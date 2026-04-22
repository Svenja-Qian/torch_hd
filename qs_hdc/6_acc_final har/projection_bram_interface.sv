//==============================================================================
// projection_bram_interface.sv
// For a single expert: generates BRAM read addresses to iterate all chunks
// of the current feature's projection row. Passes BRAM data + feat_value
// to accumulator_bank with proper alignment.
//
// 3-stage pipeline:
//   Stage 0 (S0): Issue BRAM address (o_bram_en, o_bram_addr)
//   Stage 1 (S1): BRAM read latency — data available at end of this cycle
//   Stage 2 (S2): Capture i_bram_rdata, output o_chunk_valid/data/idx
//==============================================================================

module projection_bram_interface #(
    parameter int unsigned NUM_CHUNKS   = 64,
    parameter int unsigned BRAM_ADDR_W  = 14,
    parameter int unsigned FEAT_IDX_W   = 8,
    parameter int unsigned FEAT_WIDTH   = 16,
    parameter int unsigned CHUNK_WIDTH  = 32,
    parameter int unsigned CHUNK_IDX_W  = $clog2(NUM_CHUNKS)
) (
    input  logic                            clk,
    input  logic                            rst_n,

    // Control
    input  logic                            i_enable,
    input  logic                            i_feat_valid,
    input  logic [FEAT_IDX_W-1:0]           i_feat_index,
    input  logic signed [FEAT_WIDTH-1:0]    i_feat_value,

    // BRAM interface
    output logic                            o_bram_en,
    output logic [BRAM_ADDR_W-1:0]          o_bram_addr,
    input  logic [CHUNK_WIDTH-1:0]          i_bram_rdata,

    // Output to accumulator_bank
    output logic                            o_chunk_valid,
    output logic [CHUNK_WIDTH-1:0]          o_chunk_data,
    output logic [CHUNK_IDX_W-1:0]          o_chunk_idx,
    output logic signed [FEAT_WIDTH-1:0]    o_feat_value_out,
    output logic                            o_feat_done,
    output logic                            o_busy
);

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    logic [CHUNK_IDX_W-1:0]         chunk_cnt;
    logic                           running;
    logic [BRAM_ADDR_W-1:0]         base_addr;
    logic signed [FEAT_WIDTH-1:0]   feat_val_latched;

    // Pipeline S1 registers (BRAM latency stage)
    logic                           s1_valid;
    logic [CHUNK_IDX_W-1:0]         s1_chunk_idx;
    logic signed [FEAT_WIDTH-1:0]   s1_feat_value;
    logic                           s1_last;

    // Pipeline S2 registers (output stage)
    logic                           s2_valid;
    logic [CHUNK_IDX_W-1:0]         s2_chunk_idx;
    logic signed [FEAT_WIDTH-1:0]   s2_feat_value;
    logic                           s2_last;

    assign o_busy = running;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running        <= 1'b0;
            chunk_cnt      <= '0;
            base_addr      <= '0;
            feat_val_latched <= '0;
            o_bram_en      <= 1'b0;
            o_bram_addr    <= '0;
            s1_valid       <= 1'b0;
            s1_chunk_idx   <= '0;
            s1_feat_value  <= '0;
            s1_last        <= 1'b0;
            s2_valid       <= 1'b0;
            s2_chunk_idx   <= '0;
            s2_feat_value  <= '0;
            s2_last        <= 1'b0;
            o_chunk_valid  <= 1'b0;
            o_chunk_data   <= '0;
            o_chunk_idx    <= '0;
            o_feat_value_out <= '0;
            o_feat_done    <= 1'b0;
        end else begin
            // ---- Defaults ----
            o_bram_en     <= 1'b0;
            o_chunk_valid <= 1'b0;
            o_feat_done   <= 1'b0;

            // ---- S2 → output: BRAM data now valid ----
            if (s2_valid) begin
                o_chunk_valid    <= 1'b1;
                o_chunk_data     <= i_bram_rdata;
                o_chunk_idx      <= s2_chunk_idx;
                o_feat_value_out <= s2_feat_value;
                o_feat_done      <= s2_last;
            end

            // ---- S1 → S2: just propagate ----
            s2_valid      <= s1_valid;
            s2_chunk_idx  <= s1_chunk_idx;
            s2_feat_value <= s1_feat_value;
            s2_last       <= s1_last;

            // ---- S0: issue BRAM read, feed S1 ----
            s1_valid <= 1'b0;

            if (running && i_enable) begin
                o_bram_en   <= 1'b1;
                o_bram_addr <= base_addr + BRAM_ADDR_W'(chunk_cnt);

                s1_valid      <= 1'b1;
                s1_chunk_idx  <= chunk_cnt;
                s1_feat_value <= feat_val_latched;
                s1_last       <= (chunk_cnt == CHUNK_IDX_W'(NUM_CHUNKS - 1));

                if (chunk_cnt == CHUNK_IDX_W'(NUM_CHUNKS - 1))
                    running <= 1'b0;
                else
                    chunk_cnt <= chunk_cnt + 1'b1;
            end

            // ---- New feature trigger ----
            if (i_feat_valid && i_enable && !running) begin
                running        <= 1'b1;
                chunk_cnt      <= '0;
                base_addr      <= BRAM_ADDR_W'(i_feat_index) * BRAM_ADDR_W'(NUM_CHUNKS);
                feat_val_latched <= i_feat_value;
            end
        end
    end

endmodule
