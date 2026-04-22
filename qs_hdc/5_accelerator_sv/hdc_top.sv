// =============================================================================
// Module : hdc_top
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : Top-level structural wrapper.  No compute logic here.
//          All parameters defined here; propagated downward via #().
// =============================================================================
`timescale 1ns/1ps

module hdc_top #(
    parameter int D          = 4096,
    parameter int C          = 6,
    parameter int FEAT_NUM   = 561,
    parameter int DATA_W     = 16,
    parameter int ACC_W      = 26,
    parameter int SCORE_W    = 13,
    parameter int ID_W       = 4,
    parameter int ADDR_W     = 3,
    parameter int BAUD_DIV_W = 16,
    parameter int BAUD_DIV   = 867,  // 100 MHz / 115200 - 1
    parameter CENTROID_INIT_FILE = "",
    parameter WEIGHT_INIT_FILE   = "",
    parameter bit ALLOW_DUMMY_WEIGHTS = 1'b0
)(
    input  logic clk,
    input  logic rst_n,
    input  logic rx,
    output logic tx
);

    // ── internal signals ──────────────────────────────────────────────────────
    logic [7:0]              rx_data;
    logic                    rx_valid;
    logic                    rx_error;   // available for top-level error LED etc.

    logic [FEAT_NUM*DATA_W-1:0] feature_vec;
    logic                       frame_valid;
    logic                       frame_ready;
    logic [FEAT_NUM*DATA_W-1:0] feature_buf;
    logic                       frame_pending;

    logic [D*ACC_W-1:0]         acc_out;
    logic                       proj_valid;
    logic                       proj_busy;
    logic                       proj_start;

    logic [D-1:0]               hv_out;
    logic                       hv_valid;

    logic [D-1:0]               hv_buf;
    logic                       hv_pending;
    logic                       ham_busy;
    logic                       hv_fire;
    logic [D-1:0]               hv_to_ham;

    logic [ADDR_W-1:0]          bram_rd_addr;
    logic                       bram_rd_en;
    logic [D-1:0]               bram_rd_data;

    logic [C*SCORE_W-1:0]       score_out;
    logic                       scores_valid;

    logic [ID_W-1:0]            class_id;
    logic                       result_valid;
    logic [7:0]                 tx_data_buf;
    logic                       tx_pending;

    logic                       tx_busy;

    // centroid write-port (tied off; extend with load FSM when needed)
    logic [ADDR_W-1:0] bram_wr_addr = '0;
    logic              bram_wr_en   = 1'b0;
    logic [D-1:0]      bram_wr_data = '0;

    // ── sub-module instances ──────────────────────────────────────────────────

    uart_rx #(.BAUD_DIV_W(BAUD_DIV_W)) u_uart_rx (
        .clk      (clk),
        .rst_n    (rst_n),
        .baud_div (BAUD_DIV_W'(BAUD_DIV)),
        .rx       (rx),
        .rx_data  (rx_data),
        .rx_valid (rx_valid),
        .rx_error (rx_error)
    );

    frame_ctrl #(
        .FEAT_NUM      (FEAT_NUM),
        .BYTES_PER_FEAT(2),
        .DATA_W        (DATA_W)
    ) u_frame_ctrl (
        .clk        (clk),
        .rst_n      (rst_n),
        .frame_ready(frame_ready),
        .rx_data    (rx_data),
        .rx_valid   (rx_valid),
        .feature_vec(feature_vec),
        .frame_valid(frame_valid),
        .byte_count ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feature_buf   <= '0;
            frame_pending <= 1'b0;
        end else begin
            if (frame_valid && frame_ready) begin
                feature_buf   <= feature_vec;
                frame_pending <= 1'b1;
            end
            if (proj_start) begin
                frame_pending <= 1'b0;
            end
        end
    end

    assign proj_start  = frame_pending && !proj_busy && !hv_pending && !ham_busy;
    assign frame_ready = !frame_pending && !proj_busy && !hv_pending && !ham_busy && !tx_pending;

    projection_engine #(
        .D       (D),
        .FEAT_NUM(FEAT_NUM),
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .WEIGHT_INIT_FILE(WEIGHT_INIT_FILE),
        .ALLOW_DUMMY_WEIGHTS(ALLOW_DUMMY_WEIGHTS)
    ) u_proj (
        .clk        (clk),
        .rst_n      (rst_n),
        .feature_vec(feature_buf),
        .frame_valid(proj_start),
        .acc_out    (acc_out),
        .proj_valid (proj_valid),
        .busy       (proj_busy)
    );

    binarizer #(
        .D    (D),
        .ACC_W(ACC_W)
    ) u_bin (
        .clk       (clk),
        .rst_n     (rst_n),
        .acc_in    (acc_out),
        .proj_valid(proj_valid),
        .hv_out    (hv_out),
        .hv_valid  (hv_valid)
    );

    assign hv_fire   = !ham_busy && (hv_pending || hv_valid);
    assign hv_to_ham = hv_pending ? hv_buf : hv_out;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hv_buf     <= '0;
            hv_pending <= 1'b0;
        end else begin
            if (hv_valid && ham_busy) begin
                hv_buf     <= hv_out;
                hv_pending <= 1'b1;
            end else if (hv_fire) begin
                hv_pending <= 1'b0;
            end
        end
    end

    centroid_bram #(
        .D     (D),
        .C     (C),
        .ADDR_W(ADDR_W),
        .INIT_FILE(CENTROID_INIT_FILE),
        .REQUIRE_INIT(1'b1)
    ) u_bram (
        .clk    (clk),
        .rd_addr(bram_rd_addr),
        .rd_en  (bram_rd_en),
        .rd_data(bram_rd_data),
        .wr_addr(bram_wr_addr),
        .wr_en  (bram_wr_en),
        .wr_data(bram_wr_data)
    );

    hamming_sim #(
        .D      (D),
        .C      (C),
        .ADDR_W (ADDR_W),
        .SCORE_W(SCORE_W)
    ) u_ham (
        .clk         (clk),
        .rst_n       (rst_n),
        .hv_query    (hv_to_ham),
        .hv_valid    (hv_fire),
        .rd_addr     (bram_rd_addr),
        .rd_en       (bram_rd_en),
        .centroid    (bram_rd_data),
        .score_out   (score_out),
        .scores_valid(scores_valid),
        .busy        (ham_busy)
    );

    argmax #(
        .C      (C),
        .SCORE_W(SCORE_W),
        .ID_W   (ID_W)
    ) u_argmax (
        .clk         (clk),
        .rst_n       (rst_n),
        .scores_in   (score_out),
        .scores_valid(scores_valid),
        .class_id    (class_id),
        .result_valid(result_valid)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_data_buf <= '0;
            tx_pending  <= 1'b0;
        end else begin
            if (result_valid) begin
                tx_data_buf <= {4'h0, class_id};
                tx_pending  <= 1'b1;
            end
            if (tx_pending && !tx_busy) begin
                tx_pending <= 1'b0;
            end
        end
    end

    uart_tx #(.BAUD_DIV_W(BAUD_DIV_W)) u_uart_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .baud_div (BAUD_DIV_W'(BAUD_DIV)),
        .tx_data  (tx_data_buf),
        .tx_valid (tx_pending && !tx_busy),
        .tx       (tx),
        .tx_busy  (tx_busy)
    );

endmodule
