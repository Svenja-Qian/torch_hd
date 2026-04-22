// =============================================================================
// Module : binarizer
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : hv[d] = (acc[d] > 0) ? 1 : 0  for each of D dimensions.
//          One clock latency (registered combinational).
// =============================================================================
`timescale 1ns/1ps

module binarizer #(
    parameter int D     = 4096,
    parameter int ACC_W = 26
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic [D*ACC_W-1:0] acc_in,
    input  logic               proj_valid,
    output logic [D-1:0]       hv_out,
    output logic               hv_valid
);

    // ── combinational binarise ────────────────────────────────────────────────
    logic [D-1:0] hv_comb;
    for (genvar d = 0; d < D; d++) begin : g_bin
        logic signed [ACC_W-1:0] acc_d;
        assign acc_d      = acc_in[d*ACC_W +: ACC_W];
        assign hv_comb[d] = (acc_d > 0) ? 1'b1 : 1'b0;   // Fix-③: plain 0
    end

    // ── register output + valid ───────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hv_out   <= '0;
            hv_valid <= 1'b0;
        end else begin
            hv_valid <= proj_valid;
            if (proj_valid)
                hv_out <= hv_comb;
        end
    end

endmodule
