//==============================================================================
// class_centroid_storage.sv
// Stores NUM_EXPERTS × NUM_CLASSES binary centroids, each HV_DIM bits.
// Organized as NUM_EXPERTS × NUM_CLASSES × NUM_CHUNKS × CHUNK_WIDTH.
//
// Write: 1 port for runtime loading (expert, class, chunk, data).
// Read:  SIM_PARALLELISM parallel ports — given (expert, class_base, chunk),
//        returns centroids[expert][class_base + 0..SIM_PAR-1][chunk] packed.
//
// Read timing: combinational (same-cycle). Storage is register-based
// (register file / LUTRAM). Total size: 3 × 10 × 2048 = 61,440 bits.
//
// Out-of-range read (class_base + p >= NUM_CLASSES) returns zero.
//
// o_load_done: pulses high once total writes == NUM_EXPERTS*NUM_CLASSES*NUM_CHUNKS.
//==============================================================================

module class_centroid_storage #(
    parameter int unsigned NUM_EXPERTS      = 3,
    parameter int unsigned NUM_CLASSES      = 10,
    parameter int unsigned HV_DIM           = 2048,
    parameter int unsigned CHUNK_WIDTH      = 32,
    parameter int unsigned NUM_CHUNKS       = HV_DIM / CHUNK_WIDTH,   // 64
    parameter int unsigned SIM_PARALLELISM  = 3,
    parameter int unsigned EXPERT_IDX_W     = $clog2(NUM_EXPERTS),
    parameter int unsigned CLASS_IDX_W      = $clog2(NUM_CLASSES),
    parameter int unsigned CENT_CHUNK_W     = $clog2(NUM_CHUNKS),
    // Derived
    parameter int unsigned LOAD_CNT_MAX     = NUM_EXPERTS * NUM_CLASSES * NUM_CHUNKS,
    parameter int unsigned LOAD_CNT_W       = $clog2(LOAD_CNT_MAX + 1)
) (
    input  logic                                        clk,
    input  logic                                        rst_n,

    // Write interface (runtime configuration load)
    input  logic                                        i_wr_en,
    input  logic [EXPERT_IDX_W-1:0]                     i_wr_expert_id,
    input  logic [CLASS_IDX_W-1:0]                      i_wr_class_id,
    input  logic [CENT_CHUNK_W-1:0]                     i_wr_chunk_idx,
    input  logic [CHUNK_WIDTH-1:0]                      i_wr_data,
    output logic                                        o_load_done,

    // Read interface (combinational, SIM_PARALLELISM parallel ports)
    input  logic [EXPERT_IDX_W-1:0]                     i_rd_expert,
    input  logic [CLASS_IDX_W-1:0]                      i_rd_class_base,
    input  logic [CENT_CHUNK_W-1:0]                     i_rd_chunk,
    output logic [SIM_PARALLELISM*CHUNK_WIDTH-1:0]      o_rd_data
);

    // -------------------------------------------------------------------------
    // Storage — register-based 3D array
    // -------------------------------------------------------------------------
    logic [CHUNK_WIDTH-1:0] storage [NUM_EXPERTS][NUM_CLASSES][NUM_CHUNKS];

    // -------------------------------------------------------------------------
    // Write logic
    // Note: no reset-time clearing of the 1920-entry storage array — the
    // storage is expected to be fully written via the load interface before
    // any inference begins (o_load_done indicates completion). Reset-time
    // initialization of large multi-dim arrays is not well-supported by
    // synthesis tools or Verilator.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (i_wr_en) begin
            storage[i_wr_expert_id][i_wr_class_id][i_wr_chunk_idx] <= i_wr_data;
        end
    end

    // -------------------------------------------------------------------------
    // Load-done counter
    // -------------------------------------------------------------------------
    logic [LOAD_CNT_W-1:0] load_cnt_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            load_cnt_q  <= '0;
            o_load_done <= 1'b0;
        end else begin
            if (i_wr_en && (load_cnt_q < LOAD_CNT_W'(LOAD_CNT_MAX))) begin
                load_cnt_q <= load_cnt_q + 1'b1;
                if ((load_cnt_q + 1'b1) == LOAD_CNT_W'(LOAD_CNT_MAX))
                    o_load_done <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Combinational SIM_PARALLELISM-way read with bounds check
    // -------------------------------------------------------------------------
    genvar g_p;
    generate
        for (g_p = 0; g_p < SIM_PARALLELISM; g_p++) begin : gen_rd
            logic [CLASS_IDX_W:0] class_sel_ext; // 1 extra bit for overflow check
            logic                  in_range;
            logic [CHUNK_WIDTH-1:0] data_sel;

            assign class_sel_ext = {1'b0, i_rd_class_base} + (CLASS_IDX_W+1)'(g_p);
            assign in_range      = (class_sel_ext < (CLASS_IDX_W+1)'(NUM_CLASSES));

            always_comb begin
                if (in_range)
                    data_sel = storage[i_rd_expert][class_sel_ext[CLASS_IDX_W-1:0]][i_rd_chunk];
                else
                    data_sel = '0;
            end

            assign o_rd_data[g_p*CHUNK_WIDTH +: CHUNK_WIDTH] = data_sel;
        end
    endgenerate

endmodule
