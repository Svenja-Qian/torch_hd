// =============================================================================
// Module : uart_rx
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : 8N1 UART receiver.  Samples RX at bit-centre using a run-time
//          baud_div.  Two-flop synchroniser on the async RX input.
// Fix    : Ported to SV (logic type, typedef enum, always_ff/always_comb).
// =============================================================================
`timescale 1ns/1ps

module uart_rx #(
    parameter int BAUD_DIV_W = 16
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [BAUD_DIV_W-1:0]  baud_div,
    input  logic                   rx,
    output logic [7:0]             rx_data,
    output logic                   rx_valid,
    output logic                   rx_error
);

    // ── two-flop synchroniser ─────────────────────────────────────────────────
    logic rx_s1, rx_s2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) {rx_s1, rx_s2} <= 2'b11;
        else        {rx_s1, rx_s2} <= {rx, rx_s1};
    end

    // ── FSM ───────────────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_START = 2'd1,
        S_DATA  = 2'd2,
        S_STOP  = 2'd3
    } state_t;

    state_t                  state;
    logic [BAUD_DIV_W-1:0]   baud_cnt;
    logic [2:0]              bit_idx;    // 0..7
    logic [7:0]              shift_reg;

    // half-period offset used to sample the START bit at its centre
    logic [BAUD_DIV_W-1:0] half_div;
    assign half_div = {1'b0, baud_div[BAUD_DIV_W-1:1]};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            baud_cnt  <= '0;
            bit_idx   <= '0;
            shift_reg <= '0;
            rx_data   <= '0;
            rx_valid  <= 1'b0;
            rx_error  <= 1'b0;
        end else begin
            rx_valid <= 1'b0;
            rx_error <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    if (!rx_s2) begin          // falling edge → start bit detected
                        baud_cnt <= '0;
                        state    <= S_START;
                    end
                end

                S_START: begin
                    // wait half a bit period then re-sample to reject glitches
                    if (baud_cnt == half_div) begin
                        baud_cnt <= '0;
                        if (!rx_s2) begin
                            bit_idx <= '0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE;   // glitch → abort
                        end
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                S_DATA: begin
                    if (baud_cnt == baud_div) begin
                        baud_cnt          <= '0;
                        shift_reg[bit_idx] <= rx_s2;   // LSB first
                        if (bit_idx == 3'd7) begin
                            bit_idx <= '0;
                            state   <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                S_STOP: begin
                    if (baud_cnt == baud_div) begin
                        baud_cnt <= '0;
                        if (rx_s2) begin
                            rx_data  <= shift_reg;
                            rx_valid <= 1'b1;
                        end else begin
                            rx_error <= 1'b1;
                        end
                        state <= S_IDLE;
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
