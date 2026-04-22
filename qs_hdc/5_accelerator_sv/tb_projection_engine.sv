`timescale 1ns/1ps

module tb_projection_engine;

    localparam int CLK_PERIOD = 10;
    localparam int D          = 4;
    localparam int FEAT_NUM   = 4;
    localparam int DATA_W     = 16;
    localparam int ACC_W      = 26;

    logic                       clk;
    logic                       rst_n;
    logic [FEAT_NUM*DATA_W-1:0] feature_vec;
    logic                       frame_valid;
    logic [D*ACC_W-1:0]         acc_out;
    logic                       proj_valid;
    logic                       busy;

    projection_engine #(
        .D                  (D),
        .FEAT_NUM           (FEAT_NUM),
        .DATA_W             (DATA_W),
        .ACC_W              (ACC_W),
        .ALLOW_DUMMY_WEIGHTS(1'b1)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .feature_vec(feature_vec),
        .frame_valid(frame_valid),
        .acc_out    (acc_out),
        .proj_valid (proj_valid),
        .busy       (busy)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    function automatic logic [FEAT_NUM*DATA_W-1:0] pack_features(
        input logic signed [DATA_W-1:0] f0,
        input logic signed [DATA_W-1:0] f1,
        input logic signed [DATA_W-1:0] f2,
        input logic signed [DATA_W-1:0] f3
    );
        pack_features = {f0, f1, f2, f3};
    endfunction

    function automatic logic weight_sign(input int dim_idx, input int feat_idx);
        int idx;
        begin
            idx = (dim_idx * FEAT_NUM) + feat_idx;
            weight_sign = ^idx[9:0];
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] golden_dim(
        input int dim_idx,
        input logic signed [DATA_W-1:0] f0,
        input logic signed [DATA_W-1:0] f1,
        input logic signed [DATA_W-1:0] f2,
        input logic signed [DATA_W-1:0] f3
    );
        logic signed [DATA_W-1:0] feat_arr [0:FEAT_NUM-1];
        logic signed [ACC_W-1:0] sum;
        begin
            feat_arr[0] = f0;
            feat_arr[1] = f1;
            feat_arr[2] = f2;
            feat_arr[3] = f3;
            sum = '0;
            for (int k = 0; k < FEAT_NUM; k++) begin
                if (weight_sign(dim_idx, k))
                    sum = sum + ACC_W'(feat_arr[k]);
                else
                    sum = sum - ACC_W'(feat_arr[k]);
            end
            golden_dim = sum;
        end
    endfunction

    task automatic start_frame(input logic [FEAT_NUM*DATA_W-1:0] vec);
        @(posedge clk); #1;
        feature_vec  = vec;
        frame_valid  = 1'b1;
        @(posedge clk); #1;
        frame_valid  = 1'b0;
    endtask

    task automatic wait_for_done(output int busy_cycles, output int valid_cycles);
        begin
            busy_cycles  = 0;
            valid_cycles = 0;
            while (!busy) begin
                @(posedge clk); #1;
            end
            while (busy || proj_valid) begin
                if (busy)
                    busy_cycles++;
                if (proj_valid)
                    valid_cycles++;
                @(posedge clk); #1;
            end
        end
    endtask

    task automatic check_result(
        input string test_name,
        input logic signed [DATA_W-1:0] f0,
        input logic signed [DATA_W-1:0] f1,
        input logic signed [DATA_W-1:0] f2,
        input logic signed [DATA_W-1:0] f3
    );
        logic signed [ACC_W-1:0] expected [0:D-1];
        logic signed [ACC_W-1:0] got;
        int busy_cycles;
        int valid_cycles;
        bit all_match;
        begin
            for (int d = 0; d < D; d++)
                expected[d] = golden_dim(d, f0, f1, f2, f3);

            start_frame(pack_features(f0, f1, f2, f3));
            wait_for_done(busy_cycles, valid_cycles);

            all_match = 1'b1;
            for (int d = 0; d < D; d++) begin
                got = $signed(acc_out[d*ACC_W +: ACC_W]);
                if (got !== expected[d])
                    all_match = 1'b0;
            end

            if (all_match) begin
                $display("[PASS] %s accumulator values matched expected model", test_name);
                pass_cnt++;
            end else begin
                $display("[FAIL] %s accumulator mismatch", test_name);
                for (int d = 0; d < D; d++) begin
                    got = $signed(acc_out[d*ACC_W +: ACC_W]);
                    $display("       dim %0d got=%0d expected=%0d", d, got, expected[d]);
                end
                fail_cnt++;
            end

            if (busy_cycles == (D * FEAT_NUM)) begin
                $display("[PASS] %s busy length = %0d cycles", test_name, busy_cycles);
                pass_cnt++;
            end else begin
                $display("[FAIL] %s busy length = %0d expected=%0d", test_name, busy_cycles, (D * FEAT_NUM));
                fail_cnt++;
            end

            if (valid_cycles == 1) begin
                $display("[PASS] %s proj_valid pulse width = 1", test_name);
                pass_cnt++;
            end else begin
                $display("[FAIL] %s proj_valid pulse width = %0d", test_name, valid_cycles);
                fail_cnt++;
            end
        end
    endtask

    task automatic check_busy_ignores_new_frame;
        logic [FEAT_NUM*DATA_W-1:0] first_vec;
        logic signed [ACC_W-1:0] expected_first [0:D-1];
        logic signed [ACC_W-1:0] got;
        bit all_match;
        bit saw_second_busy;
        bit saw_second_valid;
        begin
            first_vec  = pack_features(16'sd1, 16'sd2, 16'sd3, 16'sd4);

            for (int d = 0; d < D; d++)
                expected_first[d] = golden_dim(d, 16'sd1, 16'sd2, 16'sd3, 16'sd4);

            start_frame(first_vec);

            while (!busy) begin
                @(posedge clk); #1;
            end

            repeat(3) @(posedge clk);
            @(posedge clk); #1;
            // projection_engine does not latch feature_vec internally.
            // Keep feature_vec stable while busy and only pulse frame_valid here.
            feature_vec = first_vec;
            frame_valid = 1'b1;
            @(posedge clk); #1;
            frame_valid = 1'b0;

            do begin
                @(posedge clk); #1;
            end while (!proj_valid);

            all_match = 1'b1;
            for (int d = 0; d < D; d++) begin
                got = $signed(acc_out[d*ACC_W +: ACC_W]);
                if (got !== expected_first[d])
                    all_match = 1'b0;
            end

            saw_second_busy  = 1'b0;
            saw_second_valid = 1'b0;
            repeat((D * FEAT_NUM) + 2) begin
                @(posedge clk); #1;
                if (busy)
                    saw_second_busy = 1'b1;
                if (proj_valid)
                    saw_second_valid = 1'b1;
            end

            if (all_match && !saw_second_busy && !saw_second_valid) begin
                $display("[PASS] frame_valid asserted during busy is ignored");
                pass_cnt++;
            end else begin
                $display("[FAIL] busy-state frame handling mismatch");
                for (int d = 0; d < D; d++) begin
                    got = $signed(acc_out[d*ACC_W +: ACC_W]);
                    $display("       dim %0d got=%0d expected(first)=%0d", d, got, expected_first[d]);
                end
                $display("       saw_second_busy=%0d saw_second_valid=%0d", saw_second_busy, saw_second_valid);
                fail_cnt++;
            end
        end
    endtask

    initial begin
        rst_n      = 1'b0;
        feature_vec = '0;
        frame_valid = 1'b0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        check_result("mixed signed features", 16'sd1, -16'sd2, 16'sd3, -16'sd4);
        check_result("all positive features", 16'sd5, 16'sd6, 16'sd7, 16'sd8);
        check_busy_ignores_new_frame;

        $display("\n=== tb_projection_engine DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("    ALL TESTS PASSED");
        else
            $display("    *** %0d TEST(S) FAILED ***", fail_cnt);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 1000);
        $display("[TIMEOUT] tb_projection_engine exceeded maximum simulation time");
        $finish;
    end

    initial begin
        $dumpfile("tb_projection_engine.vcd");
        $dumpvars(0, tb_projection_engine);
    end

endmodule
