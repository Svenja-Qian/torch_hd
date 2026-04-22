`timescale 1ns / 1ps
module tb_best_class_selector;

    localparam int NC = 10;
    localparam int SW = 22;
    localparam int CW = 4;
    localparam int CLK_PERIOD = 10;

    logic clk, rst_n;
    logic            i_start, i_score_valid, i_last;
    logic [SW-1:0]   i_score;
    logic [CW-1:0]   i_class_idx;
    logic            o_done;
    logic [CW-1:0]   o_best_class;
    logic [SW-1:0]   o_best_score;

    best_class_selector #(.NUM_CLASSES(NC),.SCORE_WIDTH(SW)) dut(.*);

    initial clk=0;
    always #(CLK_PERIOD/2) clk=~clk;

    integer pass_cnt, fail_cnt;
    reg [SW-1:0] scores [0:NC-1];

    task automatic feed_and_check;
        input [255:0] tname; // test name (padded)
        input [CW-1:0] exp_class;
        integer i;
        begin
            // Start pulse
            @(posedge clk);
            i_start <= 1;
            @(posedge clk);
            i_start <= 0;

            // Feed scores
            for (i = 0; i < NC; i = i + 1) begin
                @(posedge clk);
                i_score_valid <= 1;
                i_score       <= scores[i];
                i_class_idx   <= i[CW-1:0];
                i_last        <= (i == NC - 1) ? 1'b1 : 1'b0;
            end
            @(posedge clk);
            i_score_valid <= 0;
            i_last        <= 0;

            // Wait for done
            repeat (3) @(posedge clk);

            if (o_done !== 1) begin
                $display("FAIL %0s: o_done not asserted", tname); fail_cnt = fail_cnt + 1;
            end else pass_cnt = pass_cnt + 1;

            if (o_best_class !== exp_class) begin
                $display("FAIL %0s: class=%0d exp=%0d score=%0d", tname, o_best_class, exp_class, o_best_score);
                fail_cnt = fail_cnt + 1;
            end else begin
                $display("PASS %0s: class=%0d score=%0d", tname, o_best_class, o_best_score);
                pass_cnt = pass_cnt + 1;
            end
        end
    endtask

    integer i;

    initial begin
        $display("=== best_class_selector Testbench ===");
        pass_cnt = 0; fail_cnt = 0;
        rst_n = 0; i_start = 0; i_score_valid = 0;
        i_score = 0; i_class_idx = 0; i_last = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // T1: Min at class 0
        for (i=0; i<NC; i=i+1) scores[i] = 100 + i*10;
        feed_and_check("T1_min_at_0", 0);

        // T2: Min at last class
        for (i=0; i<NC; i=i+1) scores[i] = 1000 - i*10;
        feed_and_check("T2_min_at_last", NC-1);

        // T3: Min at class 5
        for (i=0; i<NC; i=i+1) scores[i] = 500;
        scores[5] = 10;
        feed_and_check("T3_min_at_5", 5);

        // T4: All equal → class 0 wins
        for (i=0; i<NC; i=i+1) scores[i] = 42;
        feed_and_check("T4_all_equal", 0);

        // T5: Tie class 3 and 7 → class 3 wins
        for (i=0; i<NC; i=i+1) scores[i] = 200;
        scores[3] = 5; scores[7] = 5;
        feed_and_check("T5_tie_3_7", 3);

        // T6: Zero score at class 0
        for (i=0; i<NC; i=i+1) scores[i] = 999;
        scores[0] = 0;
        feed_and_check("T6_zero_score", 0);

        // T7: V shape, min at class 5
        for (i=0; i<NC; i=i+1) begin
            if (i <= 5) scores[i] = 100 - i*15;
            else        scores[i] = 100 - 5*15 + (i-5)*20;
        end
        feed_and_check("T7_v_shape", 5);

        // T8a/b: Consecutive runs
        for (i=0; i<NC; i=i+1) scores[i] = 300;
        scores[2] = 1;
        feed_and_check("T8a_run1", 2);
        for (i=0; i<NC; i=i+1) scores[i] = 300;
        scores[8] = 2;
        feed_and_check("T8b_run2", 8);

        // T9: Near max values
        for (i=0; i<NC; i=i+1) scores[i] = (1 << SW) - 1;
        scores[0] = (1 << SW) - 2;
        feed_and_check("T9_near_max", 0);

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
