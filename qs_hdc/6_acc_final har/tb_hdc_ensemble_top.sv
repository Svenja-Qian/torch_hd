//==============================================================================
// tb_hdc_ensemble_top.sv
// End-to-end smoke test for the HDC ensemble accelerator.
//
// Flow:
//   1. Read exported projection / centroid / boost init files from disk.
//   2. Load centroids via i_cent_wr_*.
//   3. Load boost weights.
//   4. Configure phase counts (num_shared / num_priv).
//   5. Pulse i_start, stream features in order (shared, priv0, priv1, priv2).
//   6. Wait for o_done, check o_pred_class against a software reference model.
//
// The test uses small, reproducible inputs and a simple ground-truth model
// to verify the data-path integrity rather than statistical accuracy.
//==============================================================================

`timescale 1ns / 1ps

module tb_hdc_ensemble_top;

    // -------------------------------------------------------------------------
    // DUT parameters (UCI HAR-aligned configuration)
    // -------------------------------------------------------------------------
    localparam int unsigned HV_DIM          = 2048;
    localparam int unsigned NUM_EXPERTS     = 3;
    localparam int unsigned NUM_CLASSES     = 6;
    localparam int unsigned CHUNK_WIDTH     = 32;
    localparam int unsigned FEAT_WIDTH      = 16;
    localparam int unsigned MAX_FEATURES    = 168;  // 56 shared + 112 private rows / expert
    localparam int unsigned SIM_PARALLELISM = 3;
    localparam int unsigned BOOST_WEIGHT_W  = 8;

    localparam int unsigned NUM_CHUNKS      = HV_DIM / CHUNK_WIDTH;
    localparam int unsigned FEAT_IDX_W      = $clog2(MAX_FEATURES);
    localparam int unsigned FEAT_CNT_W      = $clog2(MAX_FEATURES + 1);
    localparam int unsigned CLASS_IDX_W     = $clog2(NUM_CLASSES);
    localparam int unsigned EXPERT_IDX_W    = $clog2(NUM_EXPERTS);
    localparam int unsigned CENT_CHUNK_W    = $clog2(NUM_CHUNKS);
    localparam int unsigned HAM_WIDTH       = $clog2(HV_DIM + 1);
    localparam int unsigned FUSED_SCORE_W   = HAM_WIDTH + BOOST_WEIGHT_W + 2;

    localparam string PROJ_FILE_0           = "generated/har_hw_init/proj0.mem";
    localparam string PROJ_FILE_1           = "generated/har_hw_init/proj1.mem";
    localparam string PROJ_FILE_2           = "generated/har_hw_init/proj2.mem";
    localparam string CENT_FILE_0           = "generated/har_hw_init/centroid0.mem";
    localparam string CENT_FILE_1           = "generated/har_hw_init/centroid1.mem";
    localparam string CENT_FILE_2           = "generated/har_hw_init/centroid2.mem";
    localparam string BOOST_FILE            = "generated/har_hw_init/boost_weights.mem";

    // UCI HAR feature-importance split:
    // 561 total input features -> 56 shared + 112 private features per expert.
    localparam int unsigned N_SHARED        = 56;
    localparam int unsigned N_PRIV_0        = 112;
    localparam int unsigned N_PRIV_1        = 112;
    localparam int unsigned N_PRIV_2        = 112;
    localparam int unsigned N_TOTAL_FEATS   = N_SHARED + N_PRIV_0 + N_PRIV_1 + N_PRIV_2;

    // -------------------------------------------------------------------------
    // Clock / reset
    // -------------------------------------------------------------------------
    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;  // 100 MHz

    // -------------------------------------------------------------------------
    // Snapshot DUT accumulator values for expert 0, dims 0-7, at binarize start
    // -------------------------------------------------------------------------
    logic [31:0] dut_acc_snapshot [8];
    logic        snapshot_taken = 0;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            snapshot_taken <= 0;
        end else if (dut.binarize_start_pulse && !snapshot_taken) begin
            snapshot_taken <= 1;
            for (int b = 0; b < 8; b++)
                dut_acc_snapshot[b] <= dut.gen_expert[0].u_acc.acc[0][b];
        end
    end

    // -------------------------------------------------------------------------
    // Observability counters — count dispatches and feat_done per expert
    // -------------------------------------------------------------------------
    int dispatch_cnt [NUM_EXPERTS];
    int feat_done_cnt [NUM_EXPERTS];
    int chunk_valid_cnt [NUM_EXPERTS];
    int total_chunks_cnt [NUM_EXPERTS];
    int trace_en = 0;

    // Per-cycle trace (disabled by default; set trace_en=1 to enable)
    always_ff @(posedge clk) begin
        if (trace_en && rst_n) begin
            if (dut.sched_feat_valid) begin
                $display("[SCHED_DISPATCH t=%0t] state=%0d en=%b idx=%0d val=%0d",
                         $time, dut.state_q, dut.sched_expert_en,
                         dut.sched_feat_index, dut.sched_feat_value);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                dispatch_cnt[e]      <= 0;
                feat_done_cnt[e]     <= 0;
                chunk_valid_cnt[e]   <= 0;
                total_chunks_cnt[e]  <= 0;
            end
        end else begin
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                if (dut.sched_feat_valid && dut.sched_expert_en[e])
                    dispatch_cnt[e] <= dispatch_cnt[e] + 1;

                if (dut.pi_chunk_valid[e]) begin
                    chunk_valid_cnt[e]  <= chunk_valid_cnt[e] + 1;
                    total_chunks_cnt[e] <= total_chunks_cnt[e] + 1;
                end

                if (dut.pi_feat_done[e]) begin
                    feat_done_cnt[e] <= feat_done_cnt[e] + 1;
                    // After feat_done, verify chunks per feature
                    // (chunk_valid_cnt will be incremented THIS cycle by the last chunk,
                    //  so at this point chunk_valid_cnt+1 is the total for this feature)
                    if ((chunk_valid_cnt[e] + 1) != NUM_CHUNKS)
                        $display("[CHUNK_CHECK] *** E%0d feat_done @t=%0t chunks=%0d expected=%0d ***",
                                 e, $time, chunk_valid_cnt[e] + 1, NUM_CHUNKS);
                    chunk_valid_cnt[e] <= 0;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // DUT I/O
    // -------------------------------------------------------------------------
    logic                          i_start;
    logic                          o_done;
    logic [FEAT_CNT_W-1:0]         i_num_shared_feat;
    logic [FEAT_CNT_W-1:0]         i_num_priv_feat_0;
    logic [FEAT_CNT_W-1:0]         i_num_priv_feat_1;
    logic [FEAT_CNT_W-1:0]         i_num_priv_feat_2;
    logic                          i_feat_valid;
    logic                          o_feat_ready;
    logic signed [FEAT_WIDTH-1:0]  i_feat_value;
    logic [FEAT_IDX_W-1:0]         i_feat_index;
    logic                          i_feat_last;
    logic                          i_cent_wr_en;
    logic [EXPERT_IDX_W-1:0]       i_cent_wr_expert;
    logic [CLASS_IDX_W-1:0]        i_cent_wr_class;
    logic [CENT_CHUNK_W-1:0]       i_cent_wr_chunk;
    logic [CHUNK_WIDTH-1:0]        i_cent_wr_data;
    logic                          o_cent_load_done;
    logic [NUM_EXPERTS*BOOST_WEIGHT_W-1:0] i_boost_weight;
    logic [CLASS_IDX_W-1:0]        o_pred_class;
    logic [FUSED_SCORE_W-1:0]      o_pred_score;

    // -------------------------------------------------------------------------
    // Software "golden" reference data
    // -------------------------------------------------------------------------
    // Projection matrices: proj[expert][feat_idx][chunk] = CHUNK_WIDTH bits
    logic [CHUNK_WIDTH-1:0] proj_sw [NUM_EXPERTS][MAX_FEATURES][NUM_CHUNKS];
    logic [CHUNK_WIDTH-1:0] proj_flat_0 [0:MAX_FEATURES*NUM_CHUNKS-1];
    logic [CHUNK_WIDTH-1:0] proj_flat_1 [0:MAX_FEATURES*NUM_CHUNKS-1];
    logic [CHUNK_WIDTH-1:0] proj_flat_2 [0:MAX_FEATURES*NUM_CHUNKS-1];

    // Centroids: cent[expert][class][chunk] = CHUNK_WIDTH bits
    logic [CHUNK_WIDTH-1:0] cent_sw [NUM_EXPERTS][NUM_CLASSES][NUM_CHUNKS];
    logic [CHUNK_WIDTH-1:0] cent_flat_0 [0:NUM_CLASSES*NUM_CHUNKS-1];
    logic [CHUNK_WIDTH-1:0] cent_flat_1 [0:NUM_CLASSES*NUM_CHUNKS-1];
    logic [CHUNK_WIDTH-1:0] cent_flat_2 [0:NUM_CLASSES*NUM_CHUNKS-1];

    // Boost weights
    logic [BOOST_WEIGHT_W-1:0] boost_sw [NUM_EXPERTS];
    logic [BOOST_WEIGHT_W-1:0] boost_flat [0:NUM_EXPERTS-1];

    // Test feature stream (pre-sorted: shared, priv0, priv1, priv2)
    // Each entry: {value, index}. In the sorted stream, feat_index here
    // is the index into the projection matrix (the row address).
    int test_feat_value_arr [N_TOTAL_FEATS];
    int test_feat_index_arr [N_TOTAL_FEATS];

    // -------------------------------------------------------------------------
    // Load exported init files and build a deterministic feature stream
    // -------------------------------------------------------------------------
    initial begin
        load_hw_init_files();
        init_test_features();
    end

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    hdc_ensemble_top #(
        .HV_DIM          (HV_DIM),
        .NUM_EXPERTS     (NUM_EXPERTS),
        .NUM_CLASSES     (NUM_CLASSES),
        .CHUNK_WIDTH     (CHUNK_WIDTH),
        .FEAT_WIDTH      (FEAT_WIDTH),
        .MAX_FEATURES    (MAX_FEATURES),
        .SIM_PARALLELISM (SIM_PARALLELISM),
        .BOOST_WEIGHT_W  (BOOST_WEIGHT_W),
        .PROJ_INIT_FILE_0(PROJ_FILE_0),
        .PROJ_INIT_FILE_1(PROJ_FILE_1),
        .PROJ_INIT_FILE_2(PROJ_FILE_2)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .i_start           (i_start),
        .o_done            (o_done),
        .i_num_shared_feat (i_num_shared_feat),
        .i_num_priv_feat_0 (i_num_priv_feat_0),
        .i_num_priv_feat_1 (i_num_priv_feat_1),
        .i_num_priv_feat_2 (i_num_priv_feat_2),
        .i_feat_valid      (i_feat_valid),
        .o_feat_ready      (o_feat_ready),
        .i_feat_value      (i_feat_value),
        .i_feat_index      (i_feat_index),
        .i_feat_last       (i_feat_last),
        .i_cent_wr_en      (i_cent_wr_en),
        .i_cent_wr_expert  (i_cent_wr_expert),
        .i_cent_wr_class   (i_cent_wr_class),
        .i_cent_wr_chunk   (i_cent_wr_chunk),
        .i_cent_wr_data    (i_cent_wr_data),
        .o_cent_load_done  (o_cent_load_done),
        .i_boost_weight    (i_boost_weight),
        .o_pred_class      (o_pred_class),
        .o_pred_score      (o_pred_score)
    );

    // -------------------------------------------------------------------------
    // Tasks
    // -------------------------------------------------------------------------
    task automatic assert_file_exists(input string file_path);
        int fd;
        begin
            fd = $fopen(file_path, "r");
            if (fd == 0)
                $fatal(1, "[TB] Required init file not found: %s", file_path);
            $fclose(fd);
        end
    endtask

    task automatic load_hw_init_files;
        assert_file_exists(PROJ_FILE_0);
        assert_file_exists(PROJ_FILE_1);
        assert_file_exists(PROJ_FILE_2);
        assert_file_exists(CENT_FILE_0);
        assert_file_exists(CENT_FILE_1);
        assert_file_exists(CENT_FILE_2);
        assert_file_exists(BOOST_FILE);

        $readmemh(PROJ_FILE_0, proj_flat_0);
        $readmemh(PROJ_FILE_1, proj_flat_1);
        $readmemh(PROJ_FILE_2, proj_flat_2);
        $readmemh(CENT_FILE_0, cent_flat_0);
        $readmemh(CENT_FILE_1, cent_flat_1);
        $readmemh(CENT_FILE_2, cent_flat_2);
        $readmemh(BOOST_FILE, boost_flat);

        for (int f = 0; f < MAX_FEATURES; f++) begin
            for (int k = 0; k < NUM_CHUNKS; k++) begin
                proj_sw[0][f][k] = proj_flat_0[f*NUM_CHUNKS + k];
                proj_sw[1][f][k] = proj_flat_1[f*NUM_CHUNKS + k];
                proj_sw[2][f][k] = proj_flat_2[f*NUM_CHUNKS + k];
            end
        end

        for (int c = 0; c < NUM_CLASSES; c++) begin
            for (int k = 0; k < NUM_CHUNKS; k++) begin
                cent_sw[0][c][k] = cent_flat_0[c*NUM_CHUNKS + k];
                cent_sw[1][c][k] = cent_flat_1[c*NUM_CHUNKS + k];
                cent_sw[2][c][k] = cent_flat_2[c*NUM_CHUNKS + k];
            end
        end

        for (int e = 0; e < NUM_EXPERTS; e++)
            boost_sw[e] = boost_flat[e];

        $display("[TB] Loaded exported init files: %s, %s, %s", PROJ_FILE_0, CENT_FILE_0, BOOST_FILE);
    endtask

    task automatic init_test_features;
        // Deterministic feature sequence (pre-sorted by phase).
        // Feature indices are per-expert local row addresses inside the
        // exported projection BRAMs, not raw UCI HAR feature indices.
        for (int i = 0; i < N_SHARED; i++) begin
            test_feat_index_arr[i] = i;
            test_feat_value_arr[i] = ((i % 13) + 1) * ((i % 2) ? -5 : 7);
        end
        for (int i = 0; i < N_PRIV_0; i++) begin
            test_feat_index_arr[N_SHARED + i] = N_SHARED + i;
            test_feat_value_arr[N_SHARED + i] = ((i % 11) + 3) * ((i % 3) ? 6 : -4);
        end
        for (int i = 0; i < N_PRIV_1; i++) begin
            test_feat_index_arr[N_SHARED + N_PRIV_0 + i] = N_SHARED + i;
            test_feat_value_arr[N_SHARED + N_PRIV_0 + i] = ((i % 9) + 5) * ((i % 4) ? -3 : 8);
        end
        for (int i = 0; i < N_PRIV_2; i++) begin
            test_feat_index_arr[N_SHARED + N_PRIV_0 + N_PRIV_1 + i] = N_SHARED + i;
            test_feat_value_arr[N_SHARED + N_PRIV_0 + N_PRIV_1 + i] = ((i % 7) + 2) * ((i % 5) ? 9 : -2);
        end
        $display("[TB] Deterministic feature stream initialized.");
    endtask

    // -------------------------------------------------------------------------
    // Software reference state (module-scope; iverilog-friendly)
    // -------------------------------------------------------------------------
    logic signed [31:0] ref_acc           [NUM_EXPERTS][HV_DIM];
    logic               ref_qhv           [NUM_EXPERTS][HV_DIM];
    int                 ref_dist          [NUM_EXPERTS][NUM_CLASSES];
    int                 ref_weighted_dist [NUM_CLASSES];
    int                 ref_expected_class;

    task automatic compute_expected_class;
        int          feat_idx_list [N_TOTAL_FEATS];
        int          feat_val_list [N_TOTAL_FEATS];
        int          expert_mask   [N_TOTAL_FEATS];
        int          bit_v, d, idx, min_c;
        int          min_d;

        // Clear accumulators
        for (int e = 0; e < NUM_EXPERTS; e++)
            for (int d2 = 0; d2 < HV_DIM; d2++)
                ref_acc[e][d2] = 0;

        // Build feature list with per-feature expert masks
        for (int i = 0; i < N_SHARED; i++) begin
            feat_val_list[i] = test_feat_value_arr[i];
            feat_idx_list[i] = test_feat_index_arr[i];
            expert_mask[i]   = 3'b111;
        end
        for (int i = 0; i < N_PRIV_0; i++) begin
            feat_val_list[N_SHARED + i]    = test_feat_value_arr[N_SHARED + i];
            feat_idx_list[N_SHARED + i]    = test_feat_index_arr[N_SHARED + i];
            expert_mask[N_SHARED + i]      = 3'b001;
        end
        for (int i = 0; i < N_PRIV_1; i++) begin
            feat_val_list[N_SHARED + N_PRIV_0 + i] = test_feat_value_arr[N_SHARED + N_PRIV_0 + i];
            feat_idx_list[N_SHARED + N_PRIV_0 + i] = test_feat_index_arr[N_SHARED + N_PRIV_0 + i];
            expert_mask[N_SHARED + N_PRIV_0 + i]   = 3'b010;
        end
        for (int i = 0; i < N_PRIV_2; i++) begin
            feat_val_list[N_SHARED + N_PRIV_0 + N_PRIV_1 + i] =
                test_feat_value_arr[N_SHARED + N_PRIV_0 + N_PRIV_1 + i];
            feat_idx_list[N_SHARED + N_PRIV_0 + N_PRIV_1 + i] =
                test_feat_index_arr[N_SHARED + N_PRIV_0 + N_PRIV_1 + i];
            expert_mask[N_SHARED + N_PRIV_0 + N_PRIV_1 + i]   = 3'b100;
        end

        // Accumulate
        for (int fi = 0; fi < N_TOTAL_FEATS; fi++) begin
            idx = feat_idx_list[fi];
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                if (expert_mask[fi][e]) begin
                    for (int k = 0; k < NUM_CHUNKS; k++) begin
                        for (int b = 0; b < CHUNK_WIDTH; b++) begin
                            bit_v = proj_sw[e][idx][k][b];
                            if (bit_v)
                                ref_acc[e][k*CHUNK_WIDTH + b] += feat_val_list[fi];
                            else
                                ref_acc[e][k*CHUNK_WIDTH + b] -= feat_val_list[fi];
                        end
                    end
                end
            end
        end

        // Binarize
        for (int e = 0; e < NUM_EXPERTS; e++)
            for (int d2 = 0; d2 < HV_DIM; d2++)
                ref_qhv[e][d2] = (ref_acc[e][d2] >= 0) ? 1'b1 : 1'b0;

        // Hamming distance
        for (int e = 0; e < NUM_EXPERTS; e++) begin
            for (int c = 0; c < NUM_CLASSES; c++) begin
                d = 0;
                for (int k = 0; k < NUM_CHUNKS; k++) begin
                    for (int b = 0; b < CHUNK_WIDTH; b++) begin
                        if (ref_qhv[e][k*CHUNK_WIDTH + b] != cent_sw[e][c][k][b])
                            d++;
                    end
                end
                ref_dist[e][c] = d;
            end
        end

        // Weighted distance fusion: D[c] = sum_e alpha[e] * dist[e][c]
        for (int c = 0; c < NUM_CLASSES; c++) begin
            ref_weighted_dist[c] = 0;
            for (int e = 0; e < NUM_EXPERTS; e++)
                ref_weighted_dist[c] += int'(boost_sw[e]) * ref_dist[e][c];
        end

        // Argmin
        min_d = ref_weighted_dist[0];
        min_c = 0;
        for (int c = 1; c < NUM_CLASSES; c++) begin
            if (ref_weighted_dist[c] < min_d) begin
                min_d = ref_weighted_dist[c];
                min_c = c;
            end
        end

        $display("[TB] Expected distances per expert:");
        for (int e = 0; e < NUM_EXPERTS; e++) begin
            $write("    E%0d: ", e);
            for (int c = 0; c < NUM_CLASSES; c++) $write("%4d ", ref_dist[e][c]);
            $display("");
        end
        $display("[TB] Weighted distances:");
        for (int c = 0; c < NUM_CLASSES; c++)
            $write("%6d ", ref_weighted_dist[c]);
        $display("");
        $display("[TB] Expected argmin class = %0d (score %0d)", min_c, min_d);

        ref_expected_class = min_c;
    endtask

    task automatic load_centroids;
        i_cent_wr_en = 1'b0;
        @(posedge clk);
        for (int e = 0; e < NUM_EXPERTS; e++) begin
            for (int c = 0; c < NUM_CLASSES; c++) begin
                for (int k = 0; k < NUM_CHUNKS; k++) begin
                    i_cent_wr_en     <= 1'b1;
                    i_cent_wr_expert <= e[EXPERT_IDX_W-1:0];
                    i_cent_wr_class  <= c[CLASS_IDX_W-1:0];
                    i_cent_wr_chunk  <= k[CENT_CHUNK_W-1:0];
                    i_cent_wr_data   <= cent_sw[e][c][k];
                    @(posedge clk);
                end
            end
        end
        i_cent_wr_en <= 1'b0;
        @(posedge clk);
        $display("[TB] Centroid load complete (o_cent_load_done=%0d)", o_cent_load_done);
    endtask

    task automatic stream_features;
        // Classic valid/ready handshake:
        //   1. Drive signals with NBA (so they're stable at next edge)
        //   2. @(posedge clk) to reach the edge
        //   3. At the edge, DUT samples; if ready=1, the write was accepted
        //   4. If ready=0, hold signals (NBA keeps them) and wait more edges
        for (int fi = 0; fi < N_TOTAL_FEATS; fi++) begin
            i_feat_valid <= 1'b1;
            i_feat_value <= FEAT_WIDTH'(test_feat_value_arr[fi]);
            i_feat_index <= FEAT_IDX_W'(test_feat_index_arr[fi]);
            i_feat_last  <= (fi == N_TOTAL_FEATS - 1);
            @(posedge clk);
            // If the edge we just saw had ready=0, keep holding
            while (!o_feat_ready) @(posedge clk);
        end
        i_feat_valid <= 1'b0;
        i_feat_last  <= 1'b0;
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    int expected_class;
    int errors = 0;
    int timeout_cnt;

    initial begin
        // Default inputs
        i_start           = 0;
        i_num_shared_feat = 0;
        i_num_priv_feat_0 = 0;
        i_num_priv_feat_1 = 0;
        i_num_priv_feat_2 = 0;
        i_feat_valid      = 0;
        i_feat_value      = 0;
        i_feat_index      = 0;
        i_feat_last       = 0;
        i_cent_wr_en      = 0;
        i_cent_wr_expert  = 0;
        i_cent_wr_class   = 0;
        i_cent_wr_chunk   = 0;
        i_cent_wr_data    = 0;
        i_boost_weight    = 0;

        // Reset
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        $display("[TB] Reset released at time %0t", $time);

        // Compute expected result (uses proj_sw which is the ground truth)
        compute_expected_class();
        expected_class = ref_expected_class;

        // Load centroids
        load_centroids();

        // Load boost weights
        i_boost_weight <= {boost_sw[2], boost_sw[1], boost_sw[0]};
        @(posedge clk);

        // Set phase counts
        i_num_shared_feat <= FEAT_CNT_W'(N_SHARED);
        i_num_priv_feat_0 <= FEAT_CNT_W'(N_PRIV_0);
        i_num_priv_feat_1 <= FEAT_CNT_W'(N_PRIV_1);
        i_num_priv_feat_2 <= FEAT_CNT_W'(N_PRIV_2);
        @(posedge clk);

        // Start inference
        trace_en = 0;
        i_start <= 1'b1;
        @(posedge clk);
        i_start <= 1'b0;
        $display("[TB] Inference started at time %0t", $time);

        // Stream features in parallel (they'll go into input_buffer FIFO)
        fork
            stream_features();
        join_none

        // Wait for o_done
        timeout_cnt = 0;
        while (!o_done && timeout_cnt < 50000) begin
            @(posedge clk);
            timeout_cnt++;
        end

        if (!o_done) begin
            $display("[TB] *** TIMEOUT *** after %0d cycles", timeout_cnt);
            errors++;
        end else begin
            $display("[TB] Inference done at time %0t (took ~%0d cycles)", $time, timeout_cnt);
            $display("[TB] o_pred_class = %0d, o_pred_score = %0d", o_pred_class, o_pred_score);

            // Dump DUT's actual per-(expert,class) distances from expert_score_buffer
            $display("[TB] DUT actual distances per expert:");
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                $write("    E%0d: ", e);
                for (int c = 0; c < NUM_CLASSES; c++)
                    $write("%4d ", dut.u_esb.scores[e][c]);
                $display("");
            end
            $display("[TB] Delta (DUT - ref):");
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                $write("    E%0d: ", e);
                for (int c = 0; c < NUM_CLASSES; c++)
                    $write("%4d ", int'(dut.u_esb.scores[e][c]) - ref_dist[e][c]);
                $display("");
            end

            $display("[TB] Dispatch counts: E0=%0d E1=%0d E2=%0d (expected: E0=%0d E1=%0d E2=%0d)",
                     dispatch_cnt[0], dispatch_cnt[1], dispatch_cnt[2],
                     N_SHARED + N_PRIV_0, N_SHARED + N_PRIV_1, N_SHARED + N_PRIV_2);
            $display("[TB] feat_done counts: E0=%0d E1=%0d E2=%0d",
                     feat_done_cnt[0], feat_done_cnt[1], feat_done_cnt[2]);
            $display("[TB] Total chunks: E0=%0d E1=%0d E2=%0d (expected: %0d each)",
                     total_chunks_cnt[0], total_chunks_cnt[1], total_chunks_cnt[2],
                     (N_SHARED + N_PRIV_0) * NUM_CHUNKS);

            // Check DUT accumulator final values vs reference model. Since binarize
            // already happened, grab the DUT's query_hv and re-derive sign expectations.
            // Also dump a few raw accumulators for inspection.
            begin
                int total_bits_diff [NUM_EXPERTS];
                for (int e = 0; e < NUM_EXPERTS; e++) begin
                    total_bits_diff[e] = 0;
                    for (int k = 0; k < NUM_CHUNKS; k++) begin
                        for (int b = 0; b < CHUNK_WIDTH; b++) begin
                            if (dut.u_qhv.storage[e][k][b] != ref_qhv[e][k*CHUNK_WIDTH + b])
                                total_bits_diff[e]++;
                        end
                    end
                end
                $display("[TB] Query HV bit differences (DUT vs ref): E0=%0d E1=%0d E2=%0d (out of %0d)",
                         total_bits_diff[0], total_bits_diff[1], total_bits_diff[2], HV_DIM);

                // Sample a few bits from expert 0 chunk 0 to show value differences
                $display("[TB] Sample E0 chunk0 bits 0-7:");
                $write("    DUT qhv: ");
                for (int b = 0; b < 8; b++) $write("%b ", dut.u_qhv.storage[0][0][b]);
                $display("");
                $write("    REF qhv: ");
                for (int b = 0; b < 8; b++) $write("%b ", ref_qhv[0][b]);
                $display("");
                $display("[TB] REF acc E0 dim 0-7: %0d %0d %0d %0d %0d %0d %0d %0d",
                         ref_acc[0][0], ref_acc[0][1], ref_acc[0][2], ref_acc[0][3],
                         ref_acc[0][4], ref_acc[0][5], ref_acc[0][6], ref_acc[0][7]);
                $display("[TB] DUT acc E0 dim 0-7: %0d %0d %0d %0d %0d %0d %0d %0d",
                         $signed(dut_acc_snapshot[0]), $signed(dut_acc_snapshot[1]),
                         $signed(dut_acc_snapshot[2]), $signed(dut_acc_snapshot[3]),
                         $signed(dut_acc_snapshot[4]), $signed(dut_acc_snapshot[5]),
                         $signed(dut_acc_snapshot[6]), $signed(dut_acc_snapshot[7]));
            end

            if (int'(o_pred_class) == expected_class) begin
                $display("[TB] PASS — predicted class matches expected (%0d)", expected_class);
            end else begin
                $display("[TB] FAIL — predicted %0d, expected %0d", o_pred_class, expected_class);
                errors++;
            end
        end

        repeat (10) @(posedge clk);

        if (errors == 0)
            $display("[TB] === ALL TESTS PASSED ===");
        else
            $display("[TB] === %0d ERROR(S) ===", errors);

        $finish;
    end

    // Safety timeout
    initial begin
        #2_000_000;  // 2 ms
        $display("[TB] *** HARD TIMEOUT ***");
        $finish;
    end

endmodule
