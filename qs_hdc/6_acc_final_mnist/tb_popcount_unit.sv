//==============================================================================
// tb_popcount_unit.sv
// Testbench for popcount_unit — exhaustive edge cases + random vectors
//==============================================================================
`timescale 1ns / 1ps

module tb_popcount_unit;

    localparam int WIDTH   = 32;
    localparam int COUNT_W = $clog2(WIDTH + 1);

    logic [WIDTH-1:0]   i_data;
    logic [COUNT_W-1:0] o_count;

    popcount_unit #(.WIDTH(WIDTH)) dut (
        .i_data  (i_data),
        .o_count (o_count)
    );

    // Reference model
    function automatic int ref_popcount(input logic [WIDTH-1:0] d);
        int cnt = 0;
        for (int i = 0; i < WIDTH; i++)
            cnt += d[i];
        return cnt;
    endfunction

    int pass_cnt = 0;
    int fail_cnt = 0;
    int total    = 0;

    task automatic check(input logic [WIDTH-1:0] din);
        int expected;
        i_data = din;
        #1; // combinational settle
        expected = ref_popcount(din);
        total++;
        if (int'(o_count) !== expected) begin
            $display("FAIL: i_data=%032b  o_count=%0d  expected=%0d", din, o_count, expected);
            fail_cnt++;
        end else begin
            pass_cnt++;
        end
    endtask

    initial begin
        $display("=== popcount_unit Testbench ===");

        // Test 1: all zeros
        check(32'h0000_0000);

        // Test 2: all ones
        check(32'hFFFF_FFFF);

        // Test 3: single bit set — all 32 positions
        for (int i = 0; i < WIDTH; i++) begin
            check(32'(1) << i);
        end

        // Test 4: two bits set — first few combinations
        for (int i = 0; i < WIDTH; i++) begin
            for (int j = i + 1; j < WIDTH && j < i + 4; j++) begin
                check((32'(1) << i) | (32'(1) << j));
            end
        end

        // Test 5: walking pattern
        check(32'hAAAA_AAAA); // 16 ones
        check(32'h5555_5555); // 16 ones
        check(32'hFF00_FF00); // 16 ones
        check(32'h0F0F_0F0F); // 16 ones
        check(32'h0000_0001); // 1
        check(32'h8000_0000); // 1
        check(32'hFFFF_FFFE); // 31
        check(32'h7FFF_FFFF); // 31

        // Test 6: random vectors
        for (int i = 0; i < 10000; i++) begin
            logic [WIDTH-1:0] rnd;
            rnd = {$urandom(), $urandom()};
            rnd = rnd[WIDTH-1:0];
            check(rnd);
        end

        $display("=== Results: %0d passed, %0d failed out of %0d ===", pass_cnt, fail_cnt, total);
        if (fail_cnt == 0)
            $display("ALL TESTS PASSED");
        else
            $display("*** FAILURES DETECTED ***");

        $finish;
    end

endmodule
