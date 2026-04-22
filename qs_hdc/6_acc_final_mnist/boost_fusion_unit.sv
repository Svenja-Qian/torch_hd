//==============================================================================
// boost_fusion_unit.sv
// Computes D[c] = Σ_e α[e] × dist[e][c] for each class c.
// Reads expert_score_buffer combinationally, streams to best_class_selector.
//
// Pipeline:
//   Cycle N:   Set o_score_rd_class = c (combinational read → data available)
//   Cycle N+1: Capture data, compute D[c], output o_fused_*
//==============================================================================

module boost_fusion_unit #(
    parameter int unsigned NUM_EXPERTS    = 3,
    parameter int unsigned NUM_CLASSES    = 10,
    parameter int unsigned HAM_WIDTH      = 12,
    parameter int unsigned BOOST_WEIGHT_W = 8,
    parameter int unsigned FUSED_SCORE_W  = 22,
    parameter int unsigned CLASS_IDX_W    = $clog2(NUM_CLASSES)
) (
    input  logic                                    clk,
    input  logic                                    rst_n,

    input  logic                                    i_start,

    // Boost weights (packed)
    input  logic [NUM_EXPERTS*BOOST_WEIGHT_W-1:0]   i_boost_weight,

    // Score buffer read (combinational)
    input  logic [NUM_EXPERTS*HAM_WIDTH-1:0]        i_score_rd_dist,
    output logic [CLASS_IDX_W-1:0]                  o_score_rd_class,

    // Output to best_class_selector
    output logic                                    o_fused_valid,
    output logic [CLASS_IDX_W-1:0]                  o_fused_class,
    output logic [FUSED_SCORE_W-1:0]                o_fused_score,
    output logic                                    o_fused_last,
    output logic                                    o_done
);

    // Weight extraction
    logic [BOOST_WEIGHT_W-1:0] weights [NUM_EXPERTS];
    genvar g;
    generate
        for (g = 0; g < NUM_EXPERTS; g++) begin : gen_w
            assign weights[g] = i_boost_weight[g*BOOST_WEIGHT_W +: BOOST_WEIGHT_W];
        end
    endgenerate

    // FSM
    logic                    active;
    logic [CLASS_IDX_W-1:0]  class_cnt;

    // Read address: set combinationally so score buffer data is ready immediately.
    // On i_start, pre-set to class 0; during active, advance each cycle.
    always_comb begin
        if (active)
            o_score_rd_class = class_cnt;
        else
            o_score_rd_class = '0;
    end

    // Combinational multiply-sum using current i_score_rd_dist (which reflects o_score_rd_class)
    // But we register the result, so we compute on the p1 stage's captured data.
    // Since o_score_rd_class is combinational, i_score_rd_dist is valid same cycle.
    // We register o_score_rd_class's info into p1, and on NEXT cycle output the result.
    // But then we need to capture the dist data too.
    // 
    // Simplest correct approach: 
    //   - o_score_rd_class is driven combinationally = class_cnt
    //   - i_score_rd_dist is valid same cycle (combinational read)
    //   - We compute D[c] combinationally and register the output
    
    logic [FUSED_SCORE_W-1:0] fused_sum;
    always_comb begin
        fused_sum = '0;
        for (int e = 0; e < NUM_EXPERTS; e++)
            fused_sum = fused_sum + FUSED_SCORE_W'(weights[e]) 
                        * FUSED_SCORE_W'(i_score_rd_dist[e*HAM_WIDTH +: HAM_WIDTH]);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active        <= 1'b0;
            class_cnt     <= '0;
            o_fused_valid <= 1'b0;
            o_fused_class <= '0;
            o_fused_score <= '0;
            o_fused_last  <= 1'b0;
            o_done        <= 1'b0;
        end else begin
            // Defaults
            o_fused_valid <= 1'b0;
            o_fused_last  <= 1'b0;
            o_done        <= 1'b0;

            // Pipeline output: register the combinational result
            if (active) begin
                o_fused_valid <= 1'b1;
                o_fused_class <= class_cnt;
                o_fused_score <= fused_sum;
                o_fused_last  <= (class_cnt == CLASS_IDX_W'(NUM_CLASSES - 1));
                if (class_cnt == CLASS_IDX_W'(NUM_CLASSES - 1)) begin
                    o_done <= 1'b1;
                    active <= 1'b0;
                end else begin
                    class_cnt <= class_cnt + 1'b1;
                end
            end

            // Start trigger
            if (i_start) begin
                active    <= 1'b1;
                class_cnt <= '0;
            end
        end
    end

endmodule
