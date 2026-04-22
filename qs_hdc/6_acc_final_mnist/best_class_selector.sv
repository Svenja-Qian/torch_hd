//==============================================================================
// best_class_selector.sv
// Streaming argmin unit: receives D[c] one class per cycle from
// boost_fusion_unit, maintains running minimum, outputs pred = argmin_c D[c].
//
// Pipeline-parallel with boost_fusion_unit in S_DECISION state.
// Uses strict '<' comparison: ties broken by smallest class index.
//==============================================================================

module best_class_selector #(
    parameter int unsigned NUM_CLASSES  = 10,
    parameter int unsigned SCORE_WIDTH  = 22,
    parameter int unsigned CLASS_IDX_W  = $clog2(NUM_CLASSES)
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // Control
    input  logic                        i_start,        // Pulse: reset best_score to max

    // Streaming input (from boost_fusion_unit)
    input  logic                        i_score_valid,
    input  logic [SCORE_WIDTH-1:0]      i_score,        // D[c]
    input  logic [CLASS_IDX_W-1:0]      i_class_idx,    // c
    input  logic                        i_last,         // Last class flag

    // Output
    output logic                        o_done,         // Decision complete
    output logic [CLASS_IDX_W-1:0]      o_best_class,   // argmin class
    output logic [SCORE_WIDTH-1:0]      o_best_score    // Minimum D[c] (debug)
);

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    logic [SCORE_WIDTH-1:0]  best_score_q;
    logic [CLASS_IDX_W-1:0]  best_class_q;
    logic                    done_q;
    logic                    last_seen_q; // track that i_last was received

    // -------------------------------------------------------------------------
    // Running minimum logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            best_score_q <= '1;     // All ones = maximum
            best_class_q <= '0;
            done_q       <= 1'b0;
            last_seen_q  <= 1'b0;
        end else if (i_start) begin
            best_score_q <= '1;     // Reset to max for new decision
            best_class_q <= '0;
            done_q       <= 1'b0;
            last_seen_q  <= 1'b0;
        end else begin
            if (i_score_valid) begin
                // Strict '<': ties keep the earlier (smaller index) class
                if (i_score < best_score_q) begin
                    best_score_q <= i_score;
                    best_class_q <= i_class_idx;
                end

                // Track last
                if (i_last) begin
                    last_seen_q <= 1'b1;
                end
            end

            // Done fires one cycle after last valid score, stays high until i_start
            if (last_seen_q && !done_q) begin
                done_q <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Outputs
    // -------------------------------------------------------------------------
    assign o_done       = done_q;
    assign o_best_class = best_class_q;
    assign o_best_score = best_score_q;

endmodule
