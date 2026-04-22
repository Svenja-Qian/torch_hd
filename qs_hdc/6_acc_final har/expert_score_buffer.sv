//==============================================================================
// expert_score_buffer.sv
// Stores NUM_EXPERTS × NUM_CLASSES Hamming distance scores.
// Write: 1 port (expert, class, distance)
// Read:  given class index, outputs all experts' distances simultaneously.
//        Uses packed output for simulator compatibility.
//==============================================================================

module expert_score_buffer #(
    parameter int unsigned NUM_EXPERTS  = 3,
    parameter int unsigned NUM_CLASSES  = 10,
    parameter int unsigned HAM_WIDTH    = 12,
    parameter int unsigned EXPERT_IDX_W = $clog2(NUM_EXPERTS),
    parameter int unsigned CLASS_IDX_W  = $clog2(NUM_CLASSES)
) (
    input  logic                                    clk,
    input  logic                                    rst_n,

    // Write interface (from similarity_engine)
    input  logic                                    i_wr_en,
    input  logic [EXPERT_IDX_W-1:0]                 i_wr_expert,
    input  logic [CLASS_IDX_W-1:0]                  i_wr_class,
    input  logic [HAM_WIDTH-1:0]                    i_wr_dist,

    // Read interface — packed: o_rd_dist_packed[e*HAM_WIDTH +: HAM_WIDTH]
    input  logic [CLASS_IDX_W-1:0]                  i_rd_class,
    output logic [NUM_EXPERTS*HAM_WIDTH-1:0]        o_rd_dist_packed
);

    // Storage
    logic [HAM_WIDTH-1:0] scores [NUM_EXPERTS][NUM_CLASSES];

    // Write
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int e = 0; e < NUM_EXPERTS; e++)
                for (int c = 0; c < NUM_CLASSES; c++)
                    scores[e][c] <= '0;
        end else if (i_wr_en) begin
            scores[i_wr_expert][i_wr_class] <= i_wr_dist;
        end
    end

    // Read — combinational, packed output
    genvar g_e;
    generate
        for (g_e = 0; g_e < NUM_EXPERTS; g_e++) begin : gen_rd
            assign o_rd_dist_packed[g_e*HAM_WIDTH +: HAM_WIDTH] = scores[g_e][i_rd_class];
        end
    endgenerate

endmodule
