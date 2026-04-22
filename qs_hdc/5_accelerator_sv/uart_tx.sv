// =============================================================================
// Module : uart_tx
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : 8N1 UART transmitter.
// Fix    : S_DATA bit-output off-by-one corrected.
//          Original: tx <= shift_reg[bit_idx + 1'b1] with 4-bit bit_idx caused
//          bit[7] to be skipped (index wraps to 0 at bit_idx=7).
//          Fix: latch shift_reg at START entry, shift right each DATA cycle;
//          always drive tx from shift_reg[0] so no index arithmetic needed.
// =============================================================================
`timescale 1ns/1ps

module uart_tx #(
    parameter int BAUD_DIV_W = 16
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [BAUD_DIV_W-1:0]  baud_div,
    input  logic [7:0]             tx_data,
    input  logic                   tx_valid,
    output logic                   tx,
    output logic                   tx_busy
);

    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_START = 2'd1,
        S_DATA  = 2'd2,
        S_STOP  = 2'd3
    } state_t;

    state_t                 state;
    logic [BAUD_DIV_W-1:0]  baud_cnt;
    logic [2:0]             bit_idx;   // 0..7
    logic [7:0]             shift_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            baud_cnt  <= '0;
            bit_idx   <= '0;
            shift_reg <= '0;
            tx        <= 1'b1;   // idle high
            tx_busy   <= 1'b0;
        end else begin
            unique case (state)
                // ── idle: wait for tx_valid ───────────────────────────────────
                S_IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    if (tx_valid) begin
                        shift_reg <= tx_data;  // latch data
                        baud_cnt  <= '0;
                        tx_busy   <= 1'b1;
                        tx        <= 1'b0;     // start bit
                        state     <= S_START;
                    end
                end

                // ── start bit: hold low for one full bit period ───────────────
                S_START: begin
                    if (baud_cnt == baud_div) begin
                        baud_cnt <= '0;
                        bit_idx  <= '0;
                        tx       <= shift_reg[0];  // first data bit (bit 0)
                        state    <= S_DATA;
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                // ── data bits: LSB first, shift right each period ─────────────
                // At entry to this state tx already carries shift_reg[0].
                // Each baud_div expiry shifts right and drives the new LSB.
                S_DATA: begin
                    if (baud_cnt == baud_div) begin
                        baud_cnt  <= '0;
                        shift_reg <= shift_reg >> 1;   // shift for next bit
                        if (bit_idx == 3'd7) begin
                            // all 8 bits sent; drive stop bit
                            tx    <= 1'b1;
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                            tx      <= shift_reg[1];   // next bit after shift
                        end
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                // ── stop bit: hold high for one full bit period ───────────────
                S_STOP: begin
                    if (baud_cnt == baud_div) begin
                        baud_cnt <= '0;
                        tx_busy  <= 1'b0;
                        state    <= S_IDLE;
                    end else begin
                        baud_cnt <= baud_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
