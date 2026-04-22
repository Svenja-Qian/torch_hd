//==============================================================================
// similarity_engine.sv
// Computes Hamming distance between each expert's query_hv and each class
// centroid. Uses a 2-stage pipeline for XOR+popcount accumulation.
//
// Loops (outer → inner):
//   for expert in 0..NUM_EXPERTS-1:
//     for class_base in 0, SIM_PAR, 2*SIM_PAR, ..., < NUM_CLASSES:
//       for chunk in 0..NUM_CHUNKS-1:
//         issue read → Stage1: XOR+popcount → Stage2: ham_acc += pop
//       after 2-cycle drain, emit SIM_PAR results to expert_score_buffer
//
// Storage reads are COMBINATIONAL (query_hv_buffer and class_centroid_storage),
// so stage 0 data is available in the same cycle as the address, captured into
// Stage1 registers at end of cycle.
//
// Pipeline timing (per inner iteration chunk c):
//   Cycle 0:  addr=c driven -> data returns combinationally
//             end-of-cycle: Stage1 regs latch {query_chunk, cent_chunk[p]}
//   Cycle 1:  Stage1 computes XOR+popcount combinationally
//             end-of-cycle: Stage2 regs latch pop_result[p]
//   Cycle 2:  Stage2 adds pop_result[p] to ham_acc[p]
//
// After last chunk's address is issued, 2 drain cycles are needed.
// Then emit SIM_PAR (expert, class, dist) writes to score_buffer (1 per cycle).
//
// Out-of-range class indices (class_base + p >= NUM_CLASSES) are suppressed
// in the emit stage (i_wr_en not asserted).
//==============================================================================

module similarity_engine #(
    parameter int unsigned NUM_EXPERTS      = 3,
    parameter int unsigned NUM_CLASSES      = 10,
    parameter int unsigned CHUNK_WIDTH      = 32,
    parameter int unsigned NUM_CHUNKS       = 64,
    parameter int unsigned SIM_PARALLELISM  = 3,
    parameter int unsigned HAM_WIDTH        = 12,
    parameter int unsigned EXPERT_IDX_W     = $clog2(NUM_EXPERTS),
    parameter int unsigned CLASS_IDX_W      = $clog2(NUM_CLASSES),
    parameter int unsigned CENT_CHUNK_W     = $clog2(NUM_CHUNKS),
    parameter int unsigned POP_W            = $clog2(CHUNK_WIDTH + 1)   // 6
) (
    input  logic                                        clk,
    input  logic                                        rst_n,
    input  logic                                        i_start,

    // Query HV buffer read port (combinational)
    output logic [EXPERT_IDX_W-1:0]                     o_query_expert,
    output logic [CENT_CHUNK_W-1:0]                     o_query_chunk_idx,
    input  logic [CHUNK_WIDTH-1:0]                      i_query_chunk_data,

    // Centroid storage read port (combinational, SIM_PAR-wide)
    output logic [EXPERT_IDX_W-1:0]                     o_cent_expert,
    output logic [CLASS_IDX_W-1:0]                      o_cent_class_base,
    output logic [CENT_CHUNK_W-1:0]                     o_cent_chunk,
    input  logic [SIM_PARALLELISM*CHUNK_WIDTH-1:0]      i_cent_data,

    // Results to expert_score_buffer (1 per cycle during emit)
    output logic                                        o_result_valid,
    output logic [EXPERT_IDX_W-1:0]                     o_result_expert,
    output logic [CLASS_IDX_W-1:0]                      o_result_class,
    output logic [HAM_WIDTH-1:0]                        o_result_dist,
    output logic                                        o_all_done
);

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_COMPUTE   = 3'd1,   // issue addresses chunk=0..NUM_CHUNKS-1
        S_DRAIN     = 3'd2,   // 2 drain cycles so pipeline empties
        S_EMIT      = 3'd3,   // emit SIM_PAR results
        S_DONE      = 3'd4
    } state_t;

    state_t state_q;

    // -------------------------------------------------------------------------
    // Loop counters
    // -------------------------------------------------------------------------
    logic [EXPERT_IDX_W-1:0]     expert_q;
    logic [CLASS_IDX_W-1:0]      class_base_q;   // current class_base (0, SIM_PAR, ...)
    logic [CENT_CHUNK_W:0]       chunk_q;        // 0..NUM_CHUNKS, +1 bit for terminal
    logic [1:0]                  drain_q;        // 0..2 for drain counter
    logic [$clog2(SIM_PARALLELISM+1)-1:0] emit_q; // 0..SIM_PAR

    // -------------------------------------------------------------------------
    // Pipeline registers
    // -------------------------------------------------------------------------
    // Stage1: captured query & centroid chunks; also track validity and which
    // class_group/chunk this beat belongs to (for Stage2 demux).
    logic                             s1_valid;
    logic [CHUNK_WIDTH-1:0]           s1_query_chunk;
    logic [CHUNK_WIDTH-1:0]           s1_cent_chunk [SIM_PARALLELISM];

    // Stage2: captured popcount results
    logic                             s2_valid;
    logic [POP_W-1:0]                 s2_pop [SIM_PARALLELISM];

    // Hamming accumulators for the current class_group (SIM_PAR parallel)
    logic [HAM_WIDTH-1:0]             ham_acc [SIM_PARALLELISM];

    // -------------------------------------------------------------------------
    // Stage1 combinational: XOR + popcount (per parallel lane)
    // -------------------------------------------------------------------------
    logic [CHUNK_WIDTH-1:0]   xor_result [SIM_PARALLELISM];
    logic [POP_W-1:0]         pop_result [SIM_PARALLELISM];

    genvar g_p;
    generate
        for (g_p = 0; g_p < SIM_PARALLELISM; g_p++) begin : gen_stage1
            assign xor_result[g_p] = s1_query_chunk ^ s1_cent_chunk[g_p];
            popcount_unit #(
                .WIDTH   (CHUNK_WIDTH),
                .COUNT_W (POP_W)
            ) u_pop (
                .i_data  (xor_result[g_p]),
                .o_count (pop_result[g_p])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read address outputs (combinational; driven only during S_COMPUTE)
    // -------------------------------------------------------------------------
    always_comb begin
        o_query_expert     = expert_q;
        o_query_chunk_idx  = chunk_q[CENT_CHUNK_W-1:0];
        o_cent_expert      = expert_q;
        o_cent_class_base  = class_base_q;
        o_cent_chunk       = chunk_q[CENT_CHUNK_W-1:0];
    end

    // -------------------------------------------------------------------------
    // "issuing" flag: are we issuing a valid read this cycle?
    // -------------------------------------------------------------------------
    wire issuing = (state_q == S_COMPUTE);

    // -------------------------------------------------------------------------
    // Pre-computed extended-width arithmetic for emit logic
    // (hoisted to module scope so we don't declare variables mid-case)
    // -------------------------------------------------------------------------
    wire [CLASS_IDX_W:0] emit_class_ext =
        {1'b0, class_base_q} + (CLASS_IDX_W+1)'(emit_q);

    wire emit_class_in_range =
        (emit_class_ext < (CLASS_IDX_W+1)'(NUM_CLASSES));

    wire [CLASS_IDX_W:0] next_class_base_ext =
        {1'b0, class_base_q} + (CLASS_IDX_W+1)'(SIM_PARALLELISM);

    wire next_class_base_done =
        (next_class_base_ext >= (CLASS_IDX_W+1)'(NUM_CLASSES));

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q         <= S_IDLE;
            expert_q        <= '0;
            class_base_q    <= '0;
            chunk_q         <= '0;
            drain_q         <= '0;
            emit_q          <= '0;

            s1_valid        <= 1'b0;
            s2_valid        <= 1'b0;
            s1_query_chunk  <= '0;
            for (int p = 0; p < SIM_PARALLELISM; p++) begin
                s1_cent_chunk[p] <= '0;
                s2_pop[p]        <= '0;
                ham_acc[p]       <= '0;
            end

            o_result_valid  <= 1'b0;
            o_result_expert <= '0;
            o_result_class  <= '0;
            o_result_dist   <= '0;
            o_all_done      <= 1'b0;
        end else begin
            // Default output
            o_result_valid <= 1'b0;
            o_all_done     <= 1'b0;

            // ----------------------------------------------------------------
            // Stage1 register latch (end of Cycle N): only when issuing
            // ----------------------------------------------------------------
            s1_valid <= issuing;
            if (issuing) begin
                s1_query_chunk <= i_query_chunk_data;
                for (int p = 0; p < SIM_PARALLELISM; p++)
                    s1_cent_chunk[p] <= i_cent_data[p*CHUNK_WIDTH +: CHUNK_WIDTH];
            end

            // ----------------------------------------------------------------
            // Stage2 register latch (end of Cycle N+1): from stage1 comb
            // ----------------------------------------------------------------
            s2_valid <= s1_valid;
            if (s1_valid) begin
                for (int p = 0; p < SIM_PARALLELISM; p++)
                    s2_pop[p] <= pop_result[p];
            end

            // ----------------------------------------------------------------
            // Stage2 output: accumulate into ham_acc
            // ----------------------------------------------------------------
            if (s2_valid) begin
                for (int p = 0; p < SIM_PARALLELISM; p++)
                    ham_acc[p] <= ham_acc[p] + HAM_WIDTH'(s2_pop[p]);
            end

            // ----------------------------------------------------------------
            // FSM
            // ----------------------------------------------------------------
            case (state_q)
                S_IDLE: begin
                    if (i_start) begin
                        expert_q     <= '0;
                        class_base_q <= '0;
                        chunk_q      <= '0;
                        for (int p = 0; p < SIM_PARALLELISM; p++)
                            ham_acc[p] <= '0;
                        state_q <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // We are issuing address chunk_q this cycle.
                    // Advance chunk; when it hits NUM_CHUNKS-1, next cycle enters DRAIN.
                    if (chunk_q == CENT_CHUNK_W'(NUM_CHUNKS - 1)) begin
                        // After this cycle, pipeline still has 2 beats to drain
                        drain_q <= 2'd0;
                        state_q <= S_DRAIN;
                    end else begin
                        chunk_q <= chunk_q + 1'b1;
                    end
                end

                S_DRAIN: begin
                    // 2 drain cycles: let final s1 → s2 → ham_acc propagate.
                    // Cycle 0 of drain: s1_valid might still be 1 (from last compute
                    //                    cycle's latch). It'll latch into s2 this cycle.
                    // Cycle 1 of drain: s2_valid 1 → ham_acc updated.
                    // After drain_q reaches 1 and this cycle completes, ham_acc
                    // is fully up to date. Transition to EMIT.
                    if (drain_q == 2'd1) begin
                        emit_q  <= '0;
                        state_q <= S_EMIT;
                    end else begin
                        drain_q <= drain_q + 1'b1;
                    end
                end

                S_EMIT: begin
                    // Emit one result per cycle for SIM_PAR parallel classes.
                    // Suppress writes for out-of-range class indices.
                    if (emit_class_in_range) begin
                        o_result_valid  <= 1'b1;
                        o_result_expert <= expert_q;
                        o_result_class  <= emit_class_ext[CLASS_IDX_W-1:0];
                        o_result_dist   <= ham_acc[emit_q];
                    end

                    if (emit_q == (SIM_PARALLELISM-1)) begin
                        // Done emitting this class_group — move to next
                        // Reset per-class-group ham_acc
                        for (int p = 0; p < SIM_PARALLELISM; p++)
                            ham_acc[p] <= '0;

                        if (next_class_base_done) begin
                            // Move to next expert (or done)
                            if (expert_q == EXPERT_IDX_W'(NUM_EXPERTS - 1)) begin
                                // All done
                                state_q    <= S_DONE;
                                o_all_done <= 1'b1;
                            end else begin
                                expert_q     <= expert_q + 1'b1;
                                class_base_q <= '0;
                                chunk_q      <= '0;
                                state_q      <= S_COMPUTE;
                            end
                        end else begin
                            class_base_q <= next_class_base_ext[CLASS_IDX_W-1:0];
                            chunk_q      <= '0;
                            state_q      <= S_COMPUTE;
                        end
                    end else begin
                        emit_q <= emit_q + 1'b1;
                    end
                end

                S_DONE: begin
                    // Stay here until reset or next i_start
                    if (i_start) begin
                        expert_q     <= '0;
                        class_base_q <= '0;
                        chunk_q      <= '0;
                        for (int p = 0; p < SIM_PARALLELISM; p++)
                            ham_acc[p] <= '0;
                        state_q <= S_COMPUTE;
                    end
                end

                default: state_q <= S_IDLE;
            endcase
        end
    end

endmodule
