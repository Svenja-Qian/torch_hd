//==============================================================================
// tb_boost_fusion_unit.sv
//==============================================================================
`timescale 1ns / 1ps
module tb_boost_fusion_unit;

    localparam int NE = 3, NC = 10, HW = 12, BW = 8, FSW = 22, CIW = 4;
    localparam int CLK = 10;

    logic clk, rst_n;
    logic i_start;
    logic [NE*BW-1:0] i_boost_weight;
    logic [NE*HW-1:0] i_score_rd_dist;
    logic [CIW-1:0] o_score_rd_class;
    logic o_fused_valid, o_fused_last, o_done;
    logic [CIW-1:0] o_fused_class;
    logic [FSW-1:0] o_fused_score;

    boost_fusion_unit #(
        .NUM_EXPERTS(NE), .NUM_CLASSES(NC), .HAM_WIDTH(HW),
        .BOOST_WEIGHT_W(BW), .FUSED_SCORE_W(FSW)
    ) dut (.*);

    initial clk = 0;
    always #(CLK/2) clk = ~clk;

    // Score buffer model
    reg [HW-1:0] scores [0:NE-1][0:NC-1];

    // Provide scores combinationally based on o_score_rd_class
    always_comb begin
        for (int e = 0; e < NE; e++)
            i_score_rd_dist[e*HW +: HW] = scores[e][o_score_rd_class];
    end

    integer pass_cnt, fail_cnt;
    integer e_i, c_i;
    reg [BW-1:0] w [0:NE-1];
    reg [FSW-1:0] expected;
    integer fused_count;

    initial begin
        $display("=== boost_fusion_unit Testbench ===");
        pass_cnt = 0; fail_cnt = 0;
        rst_n = 0; i_start = 0; i_boost_weight = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Setup weights: α[0]=64 (1.0), α[1]=128 (2.0), α[2]=32 (0.5)
        // (6 fractional bits: 64 = 1.0)
        w[0] = 64; w[1] = 128; w[2] = 32;
        i_boost_weight = {w[2], w[1], w[0]};

        // Setup scores: dist[e][c] = e*10 + c
        for (e_i = 0; e_i < NE; e_i = e_i + 1)
            for (c_i = 0; c_i < NC; c_i = c_i + 1)
                scores[e_i][c_i] = e_i * 10 + c_i;

        // Start fusion
        @(negedge clk);
        i_start = 1;
        @(negedge clk);
        i_start = 0;

        // Collect results
        fused_count = 0;
        for (c_i = 0; c_i < NC + 5; c_i = c_i + 1) begin
            @(posedge clk); #1;
            if (o_fused_valid) begin
                // Expected: D[c] = Σ w[e] * scores[e][c]
                expected = 0;
                for (e_i = 0; e_i < NE; e_i = e_i + 1)
                    expected = expected + w[e_i] * scores[e_i][o_fused_class];

                if (o_fused_score !== expected) begin
                    $display("FAIL T1: class=%0d score=%0d exp=%0d",
                             o_fused_class, o_fused_score, expected);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;

                // Check class index
                if (o_fused_class !== fused_count[CIW-1:0]) begin
                    $display("FAIL T1: class order %0d exp=%0d", o_fused_class, fused_count);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;

                // Check last flag
                if (fused_count == NC - 1) begin
                    if (o_fused_last !== 1) begin
                        $display("FAIL T1: last not set"); fail_cnt = fail_cnt + 1;
                    end else pass_cnt = pass_cnt + 1;
                end

                fused_count = fused_count + 1;
            end
        end

        // T2: Should have gotten exactly NC results
        if (fused_count !== NC) begin
            $display("FAIL T2: got %0d fused results, exp %0d", fused_count, NC);
            fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;

        // T3: o_done should have been asserted
        // (already checked inline with last)

        // T4: Equal weights (all = 64 = 1.0) → simple sum
        w[0] = 64; w[1] = 64; w[2] = 64;
        i_boost_weight = {w[2], w[1], w[0]};
        for (e_i = 0; e_i < NE; e_i = e_i + 1)
            for (c_i = 0; c_i < NC; c_i = c_i + 1)
                scores[e_i][c_i] = 100;

        @(negedge clk);
        i_start = 1;
        @(negedge clk);
        i_start = 0;

        fused_count = 0;
        for (c_i = 0; c_i < NC + 5; c_i = c_i + 1) begin
            @(posedge clk); #1;
            if (o_fused_valid) begin
                // D[c] = 3 * 64 * 100 = 19200
                if (o_fused_score !== 19200) begin
                    $display("FAIL T4: class=%0d score=%0d exp=19200",
                             o_fused_class, o_fused_score);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
                fused_count = fused_count + 1;
            end
        end

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
