`timescale 1ns/1ps

module tb_binarizer;

    localparam int CLK_PERIOD = 10;
    localparam int D          = 8;
    localparam int ACC_W      = 26;

    logic                 clk;
    logic                 rst_n;
    logic [D*ACC_W-1:0]   acc_in;
    logic                 proj_valid;
    logic [D-1:0]         hv_out;
    logic                 hv_valid;

    binarizer #(.D(D), .ACC_W(ACC_W)) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .acc_in    (acc_in),
        .proj_valid(proj_valid),
        .hv_out    (hv_out),
        .hv_valid  (hv_valid)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    task automatic apply_and_check(
        input logic [D*ACC_W-1:0] acc_vec,
        input logic [D-1:0]       expected_hv,
        input string              test_name
    );
        @(posedge clk); #1;
        acc_in     = acc_vec;
        proj_valid = 1'b1;
        @(posedge clk); #1;
        proj_valid = 1'b0;

        do @(posedge clk); while (!hv_valid);
        if (hv_out === expected_hv) begin
            $display("[PASS] %s hv=%02h", test_name, hv_out);
            pass_cnt++;
        end else begin
            $display("[FAIL] %s hv=%02h expected=%02h", test_name, hv_out, expected_hv);
            fail_cnt++;
        end
    endtask

    initial begin
        logic [D*ACC_W-1:0] vec;
        int valid_cnt;

        rst_n      = 1'b0;
        acc_in     = '0;
        proj_valid = 1'b0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Test 1: all positive -> all 1
        vec = '0;
        for (int i = 0; i < D; i++)
            vec[i*ACC_W +: ACC_W] = ACC_W'(i + 1);
        apply_and_check(vec, 8'hFF, "all positive");

        // Test 2: all negative -> all 0
        vec = '0;
        for (int i = 0; i < D; i++)
            vec[i*ACC_W +: ACC_W] = -ACC_W'(i + 1);
        apply_and_check(vec, 8'h00, "all negative");

        // Test 3: all zero -> all 0
        vec = '0;
        apply_and_check(vec, 8'h00, "all zero");

        // Test 4: alternating -/+ -> 0xAA
        vec = '0;
        for (int i = 0; i < D; i++)
            vec[i*ACC_W +: ACC_W] = (i % 2 == 0) ? -26'sd1 : 26'sd1;
        apply_and_check(vec, 8'hAA, "alternating");

        // Test 5: hv_valid pulse width should be 1 cycle
        valid_cnt = 0;
        vec = '0;
        for (int i = 0; i < D; i++)
            vec[i*ACC_W +: ACC_W] = 26'sd1;

        @(posedge clk); #1;
        acc_in = vec;
        proj_valid = 1'b1;
        @(posedge clk); #1;
        proj_valid = 1'b0;

        repeat(8) begin
            @(posedge clk);
            if (hv_valid)
                valid_cnt++;
        end

        if (valid_cnt == 1) begin
            $display("[PASS] hv_valid pulse width = 1");
            pass_cnt++;
        end else begin
            $display("[FAIL] hv_valid pulse width = %0d", valid_cnt);
            fail_cnt++;
        end

        $display("\n=== tb_binarizer DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 800);
        $display("[TIMEOUT]");
        $finish;
    end

    initial begin
        $dumpfile("tb_binarizer.vcd");
        $dumpvars(0, tb_binarizer);
    end

endmodule
