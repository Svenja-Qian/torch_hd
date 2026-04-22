//==============================================================================
// tb_accumulator_bank.sv
//==============================================================================
`timescale 1ns / 1ps
module tb_accumulator_bank;

    localparam int HV_DIM     = 64;   // small for testing
    localparam int CHUNK_WIDTH = 32;
    localparam int NUM_CHUNKS  = HV_DIM / CHUNK_WIDTH; // 2
    localparam int ACC_WIDTH   = 24;
    localparam int FEAT_WIDTH  = 16;
    localparam int CHUNK_IDX_W = $clog2(NUM_CHUNKS);
    localparam int CLK = 10;

    logic clk, rst_n;
    logic i_clear, i_acc_valid, i_binarize_start;
    logic signed [FEAT_WIDTH-1:0] i_feat_value;
    logic [CHUNK_WIDTH-1:0]       i_proj_chunk;
    logic [CHUNK_IDX_W-1:0]       i_chunk_idx;
    logic o_acc_ready, o_binarize_done;
    logic o_qhv_valid;
    logic [CHUNK_IDX_W-1:0] o_qhv_chunk_idx;
    logic [CHUNK_WIDTH-1:0] o_qhv_chunk_data;

    accumulator_bank #(
        .HV_DIM(HV_DIM), .CHUNK_WIDTH(CHUNK_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FEAT_WIDTH(FEAT_WIDTH)
    ) dut (.*);

    initial clk = 0;
    always #(CLK/2) clk = ~clk;

    // Reference model
    reg signed [ACC_WIDTH-1:0] ref_acc [0:NUM_CHUNKS-1][0:CHUNK_WIDTH-1];
    integer pass_cnt, fail_cnt;
    integer e, c, b, i;

    // Helper: accumulate in reference
    task ref_accumulate(
        input signed [FEAT_WIDTH-1:0] fv,
        input [CHUNK_WIDTH-1:0] proj,
        input [CHUNK_IDX_W-1:0] cidx
    );
        integer bb;
        begin
            for (bb = 0; bb < CHUNK_WIDTH; bb = bb + 1) begin
                if (proj[bb])
                    ref_acc[cidx][bb] = ref_acc[cidx][bb] + fv;
                else
                    ref_acc[cidx][bb] = ref_acc[cidx][bb] - fv;
            end
        end
    endtask

    initial begin
        $display("=== accumulator_bank Testbench ===");
        pass_cnt = 0; fail_cnt = 0;
        rst_n = 0; i_clear = 0; i_acc_valid = 0; i_binarize_start = 0;
        i_feat_value = 0; i_proj_chunk = 0; i_chunk_idx = 0;
        for (c = 0; c < NUM_CHUNKS; c = c + 1)
            for (b = 0; b < CHUNK_WIDTH; b = b + 1)
                ref_acc[c][b] = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;

        // Wait for initial clear to finish
        wait (o_acc_ready == 1);
        @(posedge clk); #1;

        // T1: acc_ready should be 1 after clear
        if (o_acc_ready !== 1) begin
            $display("FAIL T1: not ready"); fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;

        // T2: Accumulate a single feature (value=3) with known proj_chunk
        // chunk 0: proj = 0xAAAAAAAA (alternating), chunk 1: proj = 0x55555555
        @(negedge clk);
        i_acc_valid  = 1;
        i_feat_value = 16'sd3;
        i_proj_chunk = 32'hAAAAAAAA;
        i_chunk_idx  = 0;
        ref_accumulate(16'sd3, 32'hAAAAAAAA, 0);
        @(negedge clk);
        i_proj_chunk = 32'h55555555;
        i_chunk_idx  = 1;
        ref_accumulate(16'sd3, 32'h55555555, 1);
        @(negedge clk);
        i_acc_valid = 0;
        @(posedge clk); #1;

        // T3: Accumulate second feature (value=-5) with all-ones proj
        @(negedge clk);
        i_acc_valid  = 1;
        i_feat_value = -16'sd5;
        i_proj_chunk = 32'hFFFFFFFF;
        i_chunk_idx  = 0;
        ref_accumulate(-16'sd5, 32'hFFFFFFFF, 0);
        @(negedge clk);
        i_proj_chunk = 32'hFFFFFFFF;
        i_chunk_idx  = 1;
        ref_accumulate(-16'sd5, 32'hFFFFFFFF, 1);
        @(negedge clk);
        i_acc_valid = 0;
        @(posedge clk); #1;

        // T4: Binarize and check
        @(negedge clk);
        i_binarize_start = 1;
        @(negedge clk);
        i_binarize_start = 0;

        // Collect binarized output
        for (i = 0; i < NUM_CHUNKS + 2; i = i + 1) begin
            @(posedge clk); #1;
            if (o_qhv_valid) begin
                // Check each bit
                for (b = 0; b < CHUNK_WIDTH; b = b + 1) begin
                    if ((ref_acc[o_qhv_chunk_idx][b] >= 0) !== o_qhv_chunk_data[b]) begin
                        $display("FAIL T4: chunk=%0d bit=%0d acc=%0d got=%0b exp=%0b",
                                 o_qhv_chunk_idx, b, ref_acc[o_qhv_chunk_idx][b],
                                 o_qhv_chunk_data[b], (ref_acc[o_qhv_chunk_idx][b] >= 0));
                        fail_cnt = fail_cnt + 1;
                    end else pass_cnt = pass_cnt + 1;
                end
            end
        end

        // T5: binarize_done should have fired
        // (it fires with the last qhv_valid, so check it was seen)
        // Just verify we're back to IDLE
        @(posedge clk); #1;
        if (o_acc_ready !== 1) begin
            $display("FAIL T5: not ready after binarize"); fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;

        // T6: Clear and verify accumulate again
        @(negedge clk);
        i_clear = 1;
        @(negedge clk);
        i_clear = 0;
        wait (o_acc_ready == 1);
        @(posedge clk); #1;

        // Reset ref
        for (c = 0; c < NUM_CHUNKS; c = c + 1)
            for (b = 0; b < CHUNK_WIDTH; b = b + 1)
                ref_acc[c][b] = 0;

        // Accumulate value=1 with all-zeros proj (all subtract)
        @(negedge clk);
        i_acc_valid  = 1;
        i_feat_value = 16'sd1;
        i_proj_chunk = 32'h00000000;
        i_chunk_idx  = 0;
        ref_accumulate(16'sd1, 32'h00000000, 0);
        @(negedge clk);
        i_acc_valid = 0;
        @(posedge clk);

        // Binarize - all should be 0 (acc = -1 < 0)
        @(negedge clk);
        i_binarize_start = 1;
        @(negedge clk);
        i_binarize_start = 0;

        for (i = 0; i < NUM_CHUNKS + 2; i = i + 1) begin
            @(posedge clk); #1;
            if (o_qhv_valid && o_qhv_chunk_idx == 0) begin
                if (o_qhv_chunk_data !== 32'h00000000) begin
                    $display("FAIL T6: all-sub binarize=%h exp=0", o_qhv_chunk_data);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
            end
        end

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
