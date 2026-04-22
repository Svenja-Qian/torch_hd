// =============================================================================
// Module : argmax
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : Finds the class index with the highest Hamming similarity score.
//          result_valid fires for one clock cycle after scores_valid.
// =============================================================================
`timescale 1ns/1ps

module argmax #(
    parameter int C       = 6,
    parameter int SCORE_W = 13,
    parameter int ID_W    = 4
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [C*SCORE_W-1:0]  scores_in,
    input  logic                  scores_valid,
    output logic [ID_W-1:0]       class_id,
    output logic                  result_valid
);

    logic [SCORE_W-1:0] max_val_comb;
    logic [ID_W-1:0]    max_idx_comb;

    // Pure combinational argmax; tie-breaker favors lower class index.
    always_comb begin
        max_val_comb = scores_in[0 +: SCORE_W];
        max_idx_comb = '0;
        for (int j = 1; j < C; j++) begin
            if (scores_in[j*SCORE_W +: SCORE_W] > max_val_comb) begin
                max_val_comb = scores_in[j*SCORE_W +: SCORE_W];
                max_idx_comb = ID_W'(j);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_id     <= '0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= 1'b0;

            if (scores_valid) begin
                class_id     <= max_idx_comb;
                result_valid <= 1'b1;
            end
        end
    end

endmodule
