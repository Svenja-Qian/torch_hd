//==============================================================================
// hdc_ensemble_top.sv
// Top-level HDC ensemble inference accelerator (spec v2.1).
//
// Includes an embedded top_controller_fsm (10 states), instantiates all
// sub-modules, and wires up:
//   - input_buffer        (×1)
//   - feature_scheduler   (×1)
//   - projection BRAMs    (×3, inferred from init files)
//   - projection_bram_if  (×3, generate)
//   - accumulator_bank    (×3, generate)
//   - query_hv_buffer     (×1)
//   - class_centroid_storage (×1)
//   - similarity_engine   (×1)
//   - expert_score_buffer (×1)
//   - boost_fusion_unit   (×1)
//   - best_class_selector (×1)
//
// FSM state encoding (must match parameters passed to feature_scheduler):
//   S_IDLE           = 4'd0
//   S_CLEAR_ACC      = 4'd1
//   S_SHARED_ENC     = 4'd2
//   S_PRIVATE_ENC_0  = 4'd3
//   S_PRIVATE_ENC_1  = 4'd4
//   S_PRIVATE_ENC_2  = 4'd5
//   S_BINARIZE       = 4'd6
//   S_SIMILARITY     = 4'd7
//   S_DECISION       = 4'd8
//   S_DONE           = 4'd9
//==============================================================================

module hdc_ensemble_top #(
    // Core dimensions
    parameter int unsigned HV_DIM           = 2048,
    parameter int unsigned NUM_EXPERTS      = 3,
    parameter int unsigned NUM_CLASSES      = 10,
    parameter int unsigned CHUNK_WIDTH      = 32,
    parameter int unsigned FEAT_WIDTH       = 16,
    // MNIST feature-importance partition (common_ratio=0.10, per_expert_ratio=0.25):
    // 784 total input features -> 78 shared + 118 private = 196 rows per expert.
    parameter int unsigned MAX_FEATURES     = 196,

    // Pipeline / buffer sizing
    parameter int unsigned INPUT_BUF_DEPTH  = 8,
    parameter int unsigned SIM_PARALLELISM  = 3,

    // Boost weights
    parameter int unsigned BOOST_WEIGHT_W   = 8,

    // BRAM init files (one per expert) — one .mem file of 32-bit hex words
    parameter string        PROJ_INIT_FILE_0 = "",
    parameter string        PROJ_INIT_FILE_1 = "",
    parameter string        PROJ_INIT_FILE_2 = "",

    // ---------- Derived parameters ----------
    parameter int unsigned NUM_CHUNKS    = HV_DIM / CHUNK_WIDTH,                 // 64 at HV_DIM=2048
    parameter int unsigned FEAT_IDX_W    = $clog2(MAX_FEATURES),                 // 8 at MAX_FEATURES=196
    parameter int unsigned FEAT_CNT_W    = $clog2(MAX_FEATURES + 1),             // 8 at MAX_FEATURES=196
    parameter int unsigned CLASS_IDX_W   = $clog2(NUM_CLASSES),                  // 4
    parameter int unsigned EXPERT_IDX_W  = $clog2(NUM_EXPERTS),                  // 2
    parameter int unsigned CENT_CHUNK_W  = $clog2(NUM_CHUNKS),                   // 6
    parameter int unsigned BRAM_ADDR_W   = $clog2(MAX_FEATURES * NUM_CHUNKS),    // 14
    parameter int unsigned ACC_WIDTH     = FEAT_WIDTH + $clog2(MAX_FEATURES) + 1,// 25
    parameter int unsigned HAM_WIDTH     = $clog2(HV_DIM + 1),                   // 12
    parameter int unsigned FUSED_SCORE_W = HAM_WIDTH + BOOST_WEIGHT_W + 2,       // 22
    parameter int unsigned FSM_STATE_W   = 4
) (
    input  logic                                clk,
    input  logic                                rst_n,

    // Inference control
    input  logic                                i_start,
    output logic                                o_done,

    // Phase-count configuration (set before i_start)
    input  logic [FEAT_CNT_W-1:0]               i_num_shared_feat,
    input  logic [FEAT_CNT_W-1:0]               i_num_priv_feat_0,
    input  logic [FEAT_CNT_W-1:0]               i_num_priv_feat_1,
    input  logic [FEAT_CNT_W-1:0]               i_num_priv_feat_2,

    // Feature input stream
    input  logic                                i_feat_valid,
    output logic                                o_feat_ready,
    input  logic signed [FEAT_WIDTH-1:0]        i_feat_value,
    input  logic [FEAT_IDX_W-1:0]               i_feat_index,
    input  logic                                i_feat_last,

    // Class-centroid load interface
    input  logic                                i_cent_wr_en,
    input  logic [EXPERT_IDX_W-1:0]             i_cent_wr_expert,
    input  logic [CLASS_IDX_W-1:0]              i_cent_wr_class,
    input  logic [CENT_CHUNK_W-1:0]             i_cent_wr_chunk,
    input  logic [CHUNK_WIDTH-1:0]              i_cent_wr_data,
    output logic                                o_cent_load_done,

    // Boost weight configuration
    input  logic [NUM_EXPERTS*BOOST_WEIGHT_W-1:0] i_boost_weight,

    // Prediction output
    output logic [CLASS_IDX_W-1:0]              o_pred_class,
    output logic [FUSED_SCORE_W-1:0]            o_pred_score       // debug
);

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    localparam logic [FSM_STATE_W-1:0] S_IDLE          = 4'd0;
    localparam logic [FSM_STATE_W-1:0] S_CLEAR_ACC     = 4'd1;
    localparam logic [FSM_STATE_W-1:0] S_SHARED_ENC    = 4'd2;
    localparam logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_0 = 4'd3;
    localparam logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_1 = 4'd4;
    localparam logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_2 = 4'd5;
    localparam logic [FSM_STATE_W-1:0] S_BINARIZE      = 4'd6;
    localparam logic [FSM_STATE_W-1:0] S_SIMILARITY    = 4'd7;
    localparam logic [FSM_STATE_W-1:0] S_DECISION      = 4'd8;
    localparam logic [FSM_STATE_W-1:0] S_DONE          = 4'd9;

    logic [FSM_STATE_W-1:0] state_q, state_d;

    // =========================================================================
    // Input buffer
    // =========================================================================
    logic                             ibuf_empty;
    logic                             ibuf_full;
    logic signed [FEAT_WIDTH-1:0]     ibuf_rd_value;
    logic [FEAT_IDX_W-1:0]            ibuf_rd_index;
    logic                             ibuf_rd_en;
    logic                             ibuf_wr_ready;

    input_buffer #(
        .FEAT_WIDTH (FEAT_WIDTH),
        .FEAT_IDX_W (FEAT_IDX_W),
        .DEPTH      (INPUT_BUF_DEPTH)
    ) u_input_buffer (
        .clk          (clk),
        .rst_n        (rst_n),
        .i_wr_valid   (i_feat_valid),
        .o_wr_ready   (ibuf_wr_ready),
        .i_feat_value (i_feat_value),
        .i_feat_index (i_feat_index),
        .i_feat_last  (i_feat_last),
        .i_rd_en      (ibuf_rd_en),
        .o_empty      (ibuf_empty),
        .o_full       (ibuf_full),
        .o_feat_value (ibuf_rd_value),
        .o_feat_index (ibuf_rd_index),
        .o_feat_last  ()
    );

    assign o_feat_ready = ibuf_wr_ready;

    // =========================================================================
    // Feature scheduler
    // =========================================================================
    logic [NUM_EXPERTS-1:0]       sched_expert_en;
    logic                         sched_feat_valid;
    logic signed [FEAT_WIDTH-1:0] sched_feat_value;
    logic [FEAT_IDX_W-1:0]        sched_feat_index;
    logic [NUM_EXPERTS-1:0]       sched_chunks_done;
    logic [NUM_EXPERTS-1:0]       acc_ready;
    logic                         sched_shared_done;
    logic [NUM_EXPERTS-1:0]       sched_priv_done;

    feature_scheduler #(
        .NUM_EXPERTS      (NUM_EXPERTS),
        .FEAT_WIDTH       (FEAT_WIDTH),
        .FEAT_IDX_W       (FEAT_IDX_W),
        .FEAT_CNT_W       (FEAT_CNT_W),
        .FSM_STATE_W      (FSM_STATE_W),
        .S_SHARED_ENC     (S_SHARED_ENC),
        .S_PRIVATE_ENC_0  (S_PRIVATE_ENC_0),
        .S_PRIVATE_ENC_1  (S_PRIVATE_ENC_1),
        .S_PRIVATE_ENC_2  (S_PRIVATE_ENC_2)
    ) u_scheduler (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .i_state                (state_q),
        .i_num_shared_feat      (i_num_shared_feat),
        .i_num_priv_feat_0      (i_num_priv_feat_0),
        .i_num_priv_feat_1      (i_num_priv_feat_1),
        .i_num_priv_feat_2      (i_num_priv_feat_2),
        .i_ibuf_empty           (ibuf_empty),
        .i_ibuf_feat_value      (ibuf_rd_value),
        .i_ibuf_feat_index      (ibuf_rd_index),
        .o_ibuf_rd_en           (ibuf_rd_en),
        .o_expert_en            (sched_expert_en),
        .o_feat_valid           (sched_feat_valid),
        .o_feat_value           (sched_feat_value),
        .o_feat_index           (sched_feat_index),
        .i_chunks_done          (sched_chunks_done),
        .i_acc_ready            (acc_ready),
        .o_shared_phase_done    (sched_shared_done),
        .o_private_phase_done   (sched_priv_done)
    );

    // =========================================================================
    // Per-expert: projection_bram_interface + accumulator_bank
    // Plus one BRAM per expert (inferred-style behavioral model, init file).
    // =========================================================================

    // Projection BRAM wires
    logic                         pb_en  [NUM_EXPERTS];
    logic [BRAM_ADDR_W-1:0]       pb_addr[NUM_EXPERTS];
    logic [CHUNK_WIDTH-1:0]       pb_rd  [NUM_EXPERTS];

    // Chunk outputs from projection_bram_interface → accumulator_bank
    logic                              pi_chunk_valid [NUM_EXPERTS];
    logic [CHUNK_WIDTH-1:0]            pi_chunk_data  [NUM_EXPERTS];
    logic [CENT_CHUNK_W-1:0]           pi_chunk_idx   [NUM_EXPERTS];
    logic signed [FEAT_WIDTH-1:0]      pi_feat_value  [NUM_EXPERTS];
    logic                              pi_feat_done   [NUM_EXPERTS];

    // Accumulator → query_hv_buffer
    logic                              ab_qhv_valid   [NUM_EXPERTS];
    logic [CENT_CHUNK_W-1:0]           ab_qhv_chunk   [NUM_EXPERTS];
    logic [CHUNK_WIDTH-1:0]            ab_qhv_data    [NUM_EXPERTS];
    logic                              ab_bin_done    [NUM_EXPERTS];

    // Control pulses generated by top FSM
    logic                              acc_clear_pulse;
    logic                              binarize_start_pulse;

    // ---- Projection BRAMs: one per expert, init'd from .mem file ----
    // Simple behavioral single-port BRAM (inferred by FPGA tools).
    generate
        for (genvar ge = 0; ge < NUM_EXPERTS; ge++) begin : gen_proj_bram
            // BRAM depth = MAX_FEATURES * NUM_CHUNKS (e.g., 196*64 = 12544)
            localparam int unsigned BRAM_DEPTH = MAX_FEATURES * NUM_CHUNKS;

            (* rom_style = "block" *) logic [CHUNK_WIDTH-1:0] bram_mem [0:BRAM_DEPTH-1];

            // Initialize from .mem file if provided (Vivado $readmemh).
            initial begin
                // Initialize to zero first so the ROM always has a defined driver,
                // then optionally overwrite from memory init files.
                for (int i = 0; i < BRAM_DEPTH; i++)
                    bram_mem[i] = '0;

                if (ge == 0 && PROJ_INIT_FILE_0 != "") begin
                    $readmemh(PROJ_INIT_FILE_0, bram_mem);
                end else if (ge == 1 && PROJ_INIT_FILE_1 != "") begin
                    $readmemh(PROJ_INIT_FILE_1, bram_mem);
                end else if (ge == 2 && PROJ_INIT_FILE_2 != "") begin
                    $readmemh(PROJ_INIT_FILE_2, bram_mem);
                end
            end

            // Synchronous read (1-cycle latency), matching projection_bram_interface
            // 3-stage pipeline assumption.
            always_ff @(posedge clk) begin
                if (pb_en[ge])
                    pb_rd[ge] <= bram_mem[pb_addr[ge]];
            end
        end
    endgenerate

    // ---- projection_bram_interface + accumulator_bank per expert ----
    generate
        for (genvar ge = 0; ge < NUM_EXPERTS; ge++) begin : gen_expert
            projection_bram_interface #(
                .NUM_CHUNKS   (NUM_CHUNKS),
                .BRAM_ADDR_W  (BRAM_ADDR_W),
                .FEAT_IDX_W   (FEAT_IDX_W),
                .FEAT_WIDTH   (FEAT_WIDTH),
                .CHUNK_WIDTH  (CHUNK_WIDTH)
            ) u_proj_if (
                .clk              (clk),
                .rst_n            (rst_n),
                .i_enable         (sched_expert_en[ge]),
                .i_feat_valid     (sched_feat_valid & sched_expert_en[ge]),
                .i_feat_index     (sched_feat_index),
                .i_feat_value     (sched_feat_value),
                .o_bram_en        (pb_en[ge]),
                .o_bram_addr      (pb_addr[ge]),
                .i_bram_rdata     (pb_rd[ge]),
                .o_chunk_valid    (pi_chunk_valid[ge]),
                .o_chunk_data     (pi_chunk_data[ge]),
                .o_chunk_idx      (pi_chunk_idx[ge]),
                .o_feat_value_out (pi_feat_value[ge]),
                .o_feat_done      (pi_feat_done[ge]),
                .o_busy           ()
            );

            accumulator_bank #(
                .HV_DIM      (HV_DIM),
                .CHUNK_WIDTH (CHUNK_WIDTH),
                .NUM_CHUNKS  (NUM_CHUNKS),
                .ACC_WIDTH   (ACC_WIDTH),
                .FEAT_WIDTH  (FEAT_WIDTH)
            ) u_acc (
                .clk              (clk),
                .rst_n            (rst_n),
                .i_clear          (acc_clear_pulse),
                .o_acc_ready      (acc_ready[ge]),
                .i_acc_valid      (pi_chunk_valid[ge]),
                .i_feat_value     (pi_feat_value[ge]),
                .i_proj_chunk     (pi_chunk_data[ge]),
                .i_chunk_idx      (pi_chunk_idx[ge]),
                .i_binarize_start (binarize_start_pulse),
                .o_binarize_done  (ab_bin_done[ge]),
                .o_qhv_valid      (ab_qhv_valid[ge]),
                .o_qhv_chunk_idx  (ab_qhv_chunk[ge]),
                .o_qhv_chunk_data (ab_qhv_data[ge])
            );

            // Scheduler sync: chunks_done for expert ge = pi_feat_done[ge]
            assign sched_chunks_done[ge] = pi_feat_done[ge];
        end
    endgenerate

    // =========================================================================
    // Query HV buffer
    // =========================================================================
    logic [EXPERT_IDX_W-1:0]       qhv_rd_expert;
    logic [CENT_CHUNK_W-1:0]       qhv_rd_chunk;
    logic [CHUNK_WIDTH-1:0]        qhv_rd_data;

    // Pack per-expert write ports into arrays expected by query_hv_buffer
    logic [NUM_EXPERTS-1:0]        qhv_wr_valid;
    logic [CENT_CHUNK_W-1:0]       qhv_wr_chunk [NUM_EXPERTS];
    logic [CHUNK_WIDTH-1:0]        qhv_wr_data  [NUM_EXPERTS];

    generate
        for (genvar ge = 0; ge < NUM_EXPERTS; ge++) begin : gen_qhv_pack
            assign qhv_wr_valid[ge]  = ab_qhv_valid[ge];
            assign qhv_wr_chunk[ge]  = ab_qhv_chunk[ge];
            assign qhv_wr_data[ge]   = ab_qhv_data[ge];
        end
    endgenerate

    query_hv_buffer #(
        .NUM_EXPERTS  (NUM_EXPERTS),
        .NUM_CHUNKS   (NUM_CHUNKS),
        .CHUNK_WIDTH  (CHUNK_WIDTH),
        .EXPERT_IDX_W (EXPERT_IDX_W),
        .CHUNK_IDX_W  (CENT_CHUNK_W)
    ) u_qhv (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_wr_valid     (qhv_wr_valid),
        .i_wr_chunk_idx (qhv_wr_chunk),
        .i_wr_data      (qhv_wr_data),
        .i_rd_expert    (qhv_rd_expert),
        .i_rd_chunk_idx (qhv_rd_chunk),
        .o_rd_data      (qhv_rd_data)
    );

    // =========================================================================
    // Class centroid storage
    // =========================================================================
    logic [EXPERT_IDX_W-1:0]                     cent_rd_expert;
    logic [CLASS_IDX_W-1:0]                      cent_rd_class_base;
    logic [CENT_CHUNK_W-1:0]                     cent_rd_chunk;
    logic [SIM_PARALLELISM*CHUNK_WIDTH-1:0]      cent_rd_data;

    class_centroid_storage #(
        .NUM_EXPERTS     (NUM_EXPERTS),
        .NUM_CLASSES     (NUM_CLASSES),
        .HV_DIM          (HV_DIM),
        .CHUNK_WIDTH     (CHUNK_WIDTH),
        .NUM_CHUNKS      (NUM_CHUNKS),
        .SIM_PARALLELISM (SIM_PARALLELISM),
        .EXPERT_IDX_W    (EXPERT_IDX_W),
        .CLASS_IDX_W     (CLASS_IDX_W),
        .CENT_CHUNK_W    (CENT_CHUNK_W)
    ) u_centroid (
        .clk             (clk),
        .rst_n           (rst_n),
        .i_wr_en         (i_cent_wr_en),
        .i_wr_expert_id  (i_cent_wr_expert),
        .i_wr_class_id   (i_cent_wr_class),
        .i_wr_chunk_idx  (i_cent_wr_chunk),
        .i_wr_data       (i_cent_wr_data),
        .o_load_done     (o_cent_load_done),
        .i_rd_expert     (cent_rd_expert),
        .i_rd_class_base (cent_rd_class_base),
        .i_rd_chunk      (cent_rd_chunk),
        .o_rd_data       (cent_rd_data)
    );

    // =========================================================================
    // Similarity engine
    // =========================================================================
    logic                             sim_start_pulse;
    logic                             sim_result_valid;
    logic [EXPERT_IDX_W-1:0]          sim_result_expert;
    logic [CLASS_IDX_W-1:0]           sim_result_class;
    logic [HAM_WIDTH-1:0]             sim_result_dist;
    logic                             sim_all_done;

    similarity_engine #(
        .NUM_EXPERTS     (NUM_EXPERTS),
        .NUM_CLASSES     (NUM_CLASSES),
        .CHUNK_WIDTH     (CHUNK_WIDTH),
        .NUM_CHUNKS      (NUM_CHUNKS),
        .SIM_PARALLELISM (SIM_PARALLELISM),
        .HAM_WIDTH       (HAM_WIDTH),
        .EXPERT_IDX_W    (EXPERT_IDX_W),
        .CLASS_IDX_W     (CLASS_IDX_W),
        .CENT_CHUNK_W    (CENT_CHUNK_W)
    ) u_sim (
        .clk                (clk),
        .rst_n              (rst_n),
        .i_start            (sim_start_pulse),
        .o_query_expert     (qhv_rd_expert),
        .o_query_chunk_idx  (qhv_rd_chunk),
        .i_query_chunk_data (qhv_rd_data),
        .o_cent_expert      (cent_rd_expert),
        .o_cent_class_base  (cent_rd_class_base),
        .o_cent_chunk       (cent_rd_chunk),
        .i_cent_data        (cent_rd_data),
        .o_result_valid     (sim_result_valid),
        .o_result_expert    (sim_result_expert),
        .o_result_class     (sim_result_class),
        .o_result_dist      (sim_result_dist),
        .o_all_done         (sim_all_done)
    );

    // =========================================================================
    // Expert score buffer
    // =========================================================================
    logic [CLASS_IDX_W-1:0]                 esb_rd_class;
    logic [NUM_EXPERTS*HAM_WIDTH-1:0]       esb_rd_dist_packed;

    expert_score_buffer #(
        .NUM_EXPERTS  (NUM_EXPERTS),
        .NUM_CLASSES  (NUM_CLASSES),
        .HAM_WIDTH    (HAM_WIDTH),
        .EXPERT_IDX_W (EXPERT_IDX_W),
        .CLASS_IDX_W  (CLASS_IDX_W)
    ) u_esb (
        .clk              (clk),
        .rst_n            (rst_n),
        .i_wr_en          (sim_result_valid),
        .i_wr_expert      (sim_result_expert),
        .i_wr_class       (sim_result_class),
        .i_wr_dist        (sim_result_dist),
        .i_rd_class       (esb_rd_class),
        .o_rd_dist_packed (esb_rd_dist_packed)
    );

    // =========================================================================
    // Boost fusion + best class selector (stream-parallel in S_DECISION)
    // =========================================================================
    logic                             decision_start_pulse;

    logic                             fuse_valid;
    logic [CLASS_IDX_W-1:0]           fuse_class;
    logic [FUSED_SCORE_W-1:0]         fuse_score;
    logic                             fuse_last;
    logic                             fuse_done;

    boost_fusion_unit #(
        .NUM_EXPERTS    (NUM_EXPERTS),
        .NUM_CLASSES    (NUM_CLASSES),
        .HAM_WIDTH      (HAM_WIDTH),
        .BOOST_WEIGHT_W (BOOST_WEIGHT_W),
        .FUSED_SCORE_W  (FUSED_SCORE_W),
        .CLASS_IDX_W    (CLASS_IDX_W)
    ) u_boost (
        .clk              (clk),
        .rst_n            (rst_n),
        .i_start          (decision_start_pulse),
        .i_boost_weight   (i_boost_weight),
        .i_score_rd_dist  (esb_rd_dist_packed),
        .o_score_rd_class (esb_rd_class),
        .o_fused_valid    (fuse_valid),
        .o_fused_class    (fuse_class),
        .o_fused_score    (fuse_score),
        .o_fused_last     (fuse_last),
        .o_done           (fuse_done)
    );

    logic                             best_done;
    logic [CLASS_IDX_W-1:0]           best_class;
    logic [FUSED_SCORE_W-1:0]         best_score;

    best_class_selector #(
        .NUM_CLASSES (NUM_CLASSES),
        .SCORE_WIDTH (FUSED_SCORE_W),
        .CLASS_IDX_W (CLASS_IDX_W)
    ) u_best (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (decision_start_pulse),
        .i_score_valid (fuse_valid),
        .i_score       (fuse_score),
        .i_class_idx   (fuse_class),
        .i_last        (fuse_last),
        .o_done        (best_done),
        .o_best_class  (best_class),
        .o_best_score  (best_score)
    );

    // =========================================================================
    // Top-level FSM (embedded top_controller_fsm)
    // =========================================================================

    // ---- Pulse-generation helpers (one-shot per state entry) ----
    // sim_start / decision_start: single-cycle pulses on state entry.
    logic  sim_started_q;
    logic  dec_started_q;
    logic  clear_issued_q;
    logic  binar_issued_q;

    // Track whether similarity results have all been written into the score buffer.
    // sim_all_done from similarity_engine pulses high when it transitions to S_DONE;
    // we latch it because we may consume it in the next FSM state.
    logic  sim_all_done_q;

    // ---- Next-state / control logic ----
    always_comb begin
        state_d = state_q;

        case (state_q)
            S_IDLE: begin
                if (i_start) state_d = S_CLEAR_ACC;
            end
            S_CLEAR_ACC: begin
                // After one cycle of i_clear, accumulators need NUM_CHUNKS cycles
                // to zero out; wait until all acc_ready return high.
                if (clear_issued_q && (&acc_ready)) state_d = S_SHARED_ENC;
            end
            S_SHARED_ENC: begin
                if (sched_shared_done) state_d = S_PRIVATE_ENC_0;
            end
            S_PRIVATE_ENC_0: begin
                if (sched_priv_done[0]) state_d = S_PRIVATE_ENC_1;
            end
            S_PRIVATE_ENC_1: begin
                if (sched_priv_done[1]) state_d = S_PRIVATE_ENC_2;
            end
            S_PRIVATE_ENC_2: begin
                if (sched_priv_done[2]) state_d = S_BINARIZE;
            end
            S_BINARIZE: begin
                // Wait until ALL experts have finished binarization.
                // All 3 accumulator_banks binarize in parallel, same duration.
                if (&{ab_bin_done[0], ab_bin_done[1], ab_bin_done[2]})
                    state_d = S_SIMILARITY;
            end
            S_SIMILARITY: begin
                if (sim_all_done_q) state_d = S_DECISION;
            end
            S_DECISION: begin
                if (best_done) state_d = S_DONE;
            end
            S_DONE: begin
                state_d = S_IDLE;
            end
            default: state_d = S_IDLE;
        endcase
    end

    // ---- State register + pulse-generation flags ----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q         <= S_IDLE;
            sim_started_q   <= 1'b0;
            dec_started_q   <= 1'b0;
            clear_issued_q  <= 1'b0;
            binar_issued_q  <= 1'b0;
            sim_all_done_q  <= 1'b0;
        end else begin
            // On any state change, reset the "started" latches for states we just left
            if (state_q != state_d) begin
                if (state_d != S_CLEAR_ACC)  clear_issued_q <= 1'b0;
                if (state_d != S_BINARIZE)   binar_issued_q <= 1'b0;
                if (state_d != S_SIMILARITY) sim_started_q  <= 1'b0;
                if (state_d != S_DECISION)   dec_started_q  <= 1'b0;
                if (state_d == S_SIMILARITY) sim_all_done_q <= 1'b0;
                if (state_d == S_IDLE)       sim_all_done_q <= 1'b0;
            end

            state_q <= state_d;

            // First cycle in each pulse state: raise the "issued" latch
            if (state_q == S_CLEAR_ACC  && !clear_issued_q) clear_issued_q <= 1'b1;
            if (state_q == S_BINARIZE   && !binar_issued_q) binar_issued_q <= 1'b1;
            if (state_q == S_SIMILARITY && !sim_started_q)  sim_started_q  <= 1'b1;
            if (state_q == S_DECISION   && !dec_started_q)  dec_started_q  <= 1'b1;

            // Latch sim_all_done (it's a single-cycle pulse)
            if (state_q == S_SIMILARITY && sim_all_done)
                sim_all_done_q <= 1'b1;
        end
    end

    // ---- Derived control pulses (single-cycle) ----
    assign acc_clear_pulse      = (state_q == S_CLEAR_ACC)  && !clear_issued_q;
    assign binarize_start_pulse = (state_q == S_BINARIZE)   && !binar_issued_q;
    assign sim_start_pulse      = (state_q == S_SIMILARITY) && !sim_started_q;
    assign decision_start_pulse = (state_q == S_DECISION)   && !dec_started_q;

    // =========================================================================
    // Outputs
    // =========================================================================
    assign o_done       = (state_q == S_DONE);
    assign o_pred_class = best_class;
    assign o_pred_score = best_score;

endmodule
