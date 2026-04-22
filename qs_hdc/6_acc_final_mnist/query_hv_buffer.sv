//==============================================================================
// query_hv_buffer.sv
// Centralized storage for 3 experts' binary query hypervectors.
// Organized as NUM_EXPERTS × NUM_CHUNKS × CHUNK_WIDTH bits (register-based).
//
// Write: 3 parallel write ports (one per expert), used during S_BINARIZE.
// Read:  1 combinational read port (expert, chunk_idx) → 32-bit data.
//        Used during S_SIMILARITY by similarity_engine.
//==============================================================================

module query_hv_buffer #(
    parameter int unsigned NUM_EXPERTS  = 3,
    parameter int unsigned NUM_CHUNKS   = 64,
    parameter int unsigned CHUNK_WIDTH  = 32,
    parameter int unsigned EXPERT_IDX_W = $clog2(NUM_EXPERTS),  // 2
    parameter int unsigned CHUNK_IDX_W  = $clog2(NUM_CHUNKS)    // 6
) (
    input  logic                            clk,
    input  logic                            rst_n,

    // Write interface (×NUM_EXPERTS, from accumulator_banks)
    input  logic [NUM_EXPERTS-1:0]          i_wr_valid,
    input  logic [CHUNK_IDX_W-1:0]          i_wr_chunk_idx  [NUM_EXPERTS],
    input  logic [CHUNK_WIDTH-1:0]          i_wr_data       [NUM_EXPERTS],

    // Read interface (combinational, from similarity_engine)
    input  logic [EXPERT_IDX_W-1:0]         i_rd_expert,
    input  logic [CHUNK_IDX_W-1:0]          i_rd_chunk_idx,
    output logic [CHUNK_WIDTH-1:0]          o_rd_data
);

    // -------------------------------------------------------------------------
    // Storage: NUM_EXPERTS × NUM_CHUNKS × CHUNK_WIDTH register array
    // Total: 3 × 64 × 32 = 6144 bits
    // -------------------------------------------------------------------------
    logic [CHUNK_WIDTH-1:0] storage [NUM_EXPERTS][NUM_CHUNKS];

    // -------------------------------------------------------------------------
    // Write logic — synchronous, 3 independent write ports
    // -------------------------------------------------------------------------
    genvar g_e;
    generate
        for (g_e = 0; g_e < NUM_EXPERTS; g_e++) begin : gen_wr
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int c = 0; c < NUM_CHUNKS; c++)
                        storage[g_e][c] <= '0;
                end else if (i_wr_valid[g_e]) begin
                    storage[g_e][i_wr_chunk_idx[g_e]] <= i_wr_data[g_e];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read logic — combinational (same-cycle output)
    // -------------------------------------------------------------------------
    assign o_rd_data = storage[i_rd_expert][i_rd_chunk_idx];

endmodule
