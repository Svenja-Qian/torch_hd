//==============================================================================
// tb_input_buffer.sv
// Testbench for input_buffer — FIFO behaviour verification
//==============================================================================
`timescale 1ns / 1ps

module tb_input_buffer;

    localparam int FEAT_WIDTH  = 16;
    localparam int FEAT_IDX_W  = 8;
    localparam int DEPTH       = 8;
    localparam int CLK_PERIOD  = 10;

    logic                        clk, rst_n;
    logic                        i_wr_valid, o_wr_ready;
    logic signed [FEAT_WIDTH-1:0] i_feat_value;
    logic [FEAT_IDX_W-1:0]       i_feat_index;
    logic                        i_feat_last;
    logic                        i_rd_en, o_empty, o_full;
    logic signed [FEAT_WIDTH-1:0] o_feat_value;
    logic [FEAT_IDX_W-1:0]       o_feat_index;
    logic                        o_feat_last;

    input_buffer #(
        .FEAT_WIDTH (FEAT_WIDTH),
        .FEAT_IDX_W (FEAT_IDX_W),
        .DEPTH      (DEPTH)
    ) dut (.*);

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    // Reference queues
    logic signed [FEAT_WIDTH-1:0] ref_val [$];
    logic [FEAT_IDX_W-1:0]       ref_idx [$];
    logic                        ref_last[$];

    initial begin
        $display("=== input_buffer Testbench ===");
        rst_n = 0; i_wr_valid = 0; i_rd_en = 0;
        i_feat_value = '0; i_feat_index = '0; i_feat_last = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk); #1;

        // ----- T1: Empty after reset -----
        if (o_empty !== 1) begin $error("T1: not empty"); fail_cnt++; end else pass_cnt++;
        if (o_full  !== 0) begin $error("T1: full");      fail_cnt++; end else pass_cnt++;

        // ----- T2: Write DEPTH entries -----
        for (int i = 0; i < DEPTH; i++) begin
            @(posedge clk);
            i_wr_valid   <= 1;
            i_feat_value <= FEAT_WIDTH'(i * 3 - 100);
            i_feat_index <= FEAT_IDX_W'(i);
            i_feat_last  <= (i == DEPTH - 1);
        end
        @(posedge clk);
        i_wr_valid <= 0;
        @(posedge clk); #1;

        if (o_full  !== 1) begin $error("T2: not full");  fail_cnt++; end else pass_cnt++;
        if (o_empty !== 0) begin $error("T2: empty");     fail_cnt++; end else pass_cnt++;

        // ----- T3: Read all back, check FIFO order -----
        // Output is combinational from mem[rd_ptr]. Check BEFORE issuing rd_en.
        for (int i = 0; i < DEPTH; i++) begin
            // Check output at head (combinational)
            #1;
            if (o_feat_value !== FEAT_WIDTH'(i * 3 - 100)) begin
                $error("T3[%0d]: val=%0d exp=%0d", i, o_feat_value, FEAT_WIDTH'(i*3-100));
                fail_cnt++;
            end else pass_cnt++;
            if (o_feat_index !== FEAT_IDX_W'(i)) begin
                $error("T3[%0d]: idx=%0d exp=%0d", i, o_feat_index, i);
                fail_cnt++;
            end else pass_cnt++;
            if (o_feat_last !== (i == DEPTH - 1)) begin
                $error("T3[%0d]: last=%0b exp=%0b", i, o_feat_last, (i == DEPTH-1));
                fail_cnt++;
            end else pass_cnt++;
            // Issue read to advance pointer
            @(posedge clk);
            i_rd_en <= 1;
            @(posedge clk);
            i_rd_en <= 0;
        end
        #1;
        if (o_empty !== 1) begin $error("T3: not empty"); fail_cnt++; end else pass_cnt++;

        // ----- T4: Simultaneous read+write -----
        // Write 4 entries
        for (int i = 0; i < 4; i++) begin
            @(posedge clk);
            i_wr_valid   <= 1;
            i_feat_value <= FEAT_WIDTH'(i + 50);
            i_feat_index <= FEAT_IDX_W'(i + 10);
            i_feat_last  <= 0;
        end
        @(posedge clk); i_wr_valid <= 0;
        @(posedge clk); #1;

        // Simultaneous R+W for 4 cycles
        for (int i = 0; i < 4; i++) begin
            @(posedge clk);
            i_wr_valid   <= 1;
            i_feat_value <= FEAT_WIDTH'(i + 100);
            i_feat_index <= FEAT_IDX_W'(i + 20);
            i_feat_last  <= 0;
            i_rd_en      <= 1;
        end
        @(posedge clk);
        i_wr_valid <= 0; i_rd_en <= 0;
        @(posedge clk); #1;

        if (o_empty !== 0) begin $error("T4: unexpected empty"); fail_cnt++; end else pass_cnt++;
        // Drain
        for (int i = 0; i < 4; i++) begin
            @(posedge clk); i_rd_en <= 1; @(posedge clk); i_rd_en <= 0;
        end
        @(posedge clk); #1;
        if (o_empty !== 1) begin $error("T4: not empty after drain"); fail_cnt++; end else pass_cnt++;

        // ----- T5: Overflow protection -----
        for (int i = 0; i < DEPTH; i++) begin
            @(posedge clk);
            i_wr_valid   <= 1;
            i_feat_value <= FEAT_WIDTH'(i);
            i_feat_index <= FEAT_IDX_W'(i);
            i_feat_last  <= 0;
        end
        @(posedge clk); i_wr_valid <= 0;
        @(posedge clk); #1;
        if (o_full !== 1) begin $error("T5: not full"); fail_cnt++; end else pass_cnt++;

        // Try to write when full
        @(posedge clk);
        i_wr_valid   <= 1;
        i_feat_value <= FEAT_WIDTH'(99);
        i_feat_index <= FEAT_IDX_W'(99);
        i_feat_last  <= 1;
        #1;
        // o_wr_ready should be 0 since full
        @(posedge clk); #1;
        i_wr_valid <= 0;
        // Head should still be entry 0
        if (o_feat_value !== FEAT_WIDTH'(0)) begin
            $error("T5: data corruption, head=%0d exp=0", o_feat_value);
            fail_cnt++;
        end else pass_cnt++;

        // Cleanup
        for (int i = 0; i < DEPTH; i++) begin
            @(posedge clk); i_rd_en <= 1; @(posedge clk); i_rd_en <= 0;
        end

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end

endmodule
