// =============================================================================
// Module : frame_ctrl
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : Reassembles UART byte stream into a FEAT_NUM×DATA_W feature vector.
//          Frame format: [SOF=0xAA] [FEAT_NUM×2 bytes, big-endian] [EOF=0x55]
// =============================================================================
`timescale 1ns/1ps

module frame_ctrl #(
    parameter int FEAT_NUM       = 561,
    parameter int BYTES_PER_FEAT = 2,
    parameter int DATA_W         = 16
)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           frame_ready,
    input  logic [7:0]                     rx_data,
    input  logic                           rx_valid,
    output logic [FEAT_NUM*DATA_W-1:0]     feature_vec,
    output logic                           frame_valid,
    output logic [10:0]                    byte_count
);

    localparam int TOTAL_BYTES = FEAT_NUM * BYTES_PER_FEAT;
    localparam int CNT_W       = (TOTAL_BYTES <= 1) ? 1 : $clog2(TOTAL_BYTES);

    typedef enum logic [1:0] {
        S_WAIT_SOF = 2'd0,
        S_DATA     = 2'd1,
        S_WAIT_EOF = 2'd2
    } state_t;

    state_t                       state;
    logic [CNT_W-1:0]             bcnt;
    logic [7:0]                   tmp_hi;
    logic [$clog2(FEAT_NUM)-1:0]  feat_idx;

    // feat_idx = bcnt >> 1  (which feature pair we are filling)
    assign feat_idx = bcnt[CNT_W-1:1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_WAIT_SOF;
            bcnt        <= '0;
            byte_count  <= '0;
            frame_valid <= 1'b0;
            feature_vec <= '0;
            tmp_hi      <= '0;
        end else begin
            frame_valid <= 1'b0;

            if (rx_valid) begin
                unique case (state)
                    // ── wait for SOF ──────────────────────────────────────────
                    S_WAIT_SOF: begin
                        if (rx_data == 8'hAA) begin
                            if (frame_ready) begin
                                feature_vec <= '0;
                                tmp_hi      <= '0;
                                byte_count  <= '0;
                                bcnt        <= '0;
                                state       <= S_DATA;
                            end
                        end
                    end

                    // ── collect data bytes ────────────────────────────────────
                    S_DATA: begin
                        byte_count <= bcnt;

                        if (bcnt[0] == 1'b0) begin
                            // even: high byte
                            tmp_hi <= rx_data;
                        end else begin
                            // odd: low byte; write full word to correct slot
                            // feature[0] at MSB end: slot = (FEAT_NUM-1-feat_idx)
                            feature_vec[(FEAT_NUM-1-feat_idx)*DATA_W +: DATA_W]
                                <= {tmp_hi, rx_data};
                        end

                        if (bcnt == CNT_W'(TOTAL_BYTES - 1)) begin
                            bcnt  <= '0;
                            state <= S_WAIT_EOF;
                        end else begin
                            bcnt <= bcnt + 1'b1;
                        end
                    end

                    // ── wait for EOF ──────────────────────────────────────────
                    S_WAIT_EOF: begin
                        if (rx_data == 8'h55)
                            frame_valid <= 1'b1;
                        state <= S_WAIT_SOF;
                    end

                    default: state <= S_WAIT_SOF;
                endcase
            end
        end
    end

endmodule

