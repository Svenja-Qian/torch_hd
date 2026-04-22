`timescale 1ns / 1ps
module tb_query_hv_buffer;

    localparam int NUM_EXPERTS  = 3;
    localparam int NUM_CHUNKS   = 64;
    localparam int CHUNK_WIDTH  = 32;
    localparam int EXPERT_IDX_W = 2;
    localparam int CHUNK_IDX_W  = 6;
    localparam int CLK_PERIOD   = 10;

    logic clk, rst_n;
    logic [NUM_EXPERTS-1:0]   i_wr_valid;
    logic [CHUNK_IDX_W-1:0]   i_wr_chunk_idx [NUM_EXPERTS];
    logic [CHUNK_WIDTH-1:0]   i_wr_data      [NUM_EXPERTS];
    logic [EXPERT_IDX_W-1:0]  i_rd_expert;
    logic [CHUNK_IDX_W-1:0]   i_rd_chunk_idx;
    logic [CHUNK_WIDTH-1:0]   o_rd_data;

    query_hv_buffer #(
        .NUM_EXPERTS(NUM_EXPERTS), .NUM_CHUNKS(NUM_CHUNKS), .CHUNK_WIDTH(CHUNK_WIDTH)
    ) dut (.*);

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [CHUNK_WIDTH-1:0] ref_mem [0:NUM_EXPERTS-1][0:NUM_CHUNKS-1];
    integer pass_cnt, fail_cnt;
    integer e, c;

    initial begin
        $display("=== query_hv_buffer Testbench ===");
        pass_cnt = 0; fail_cnt = 0;
        rst_n = 0; i_wr_valid = 0;
        for (e = 0; e < NUM_EXPERTS; e = e + 1) begin
            i_wr_chunk_idx[e] = 0; i_wr_data[e] = 0;
        end
        i_rd_expert = 0; i_rd_chunk_idx = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // T1: Reset check
        for (e = 0; e < NUM_EXPERTS; e = e + 1) begin
            for (c = 0; c < 4; c = c + 1) begin
                i_rd_expert    = e[EXPERT_IDX_W-1:0];
                i_rd_chunk_idx = c[CHUNK_IDX_W-1:0];
                #1;
                if (o_rd_data !== 0) begin
                    $display("FAIL T1: [%0d][%0d]=%h", e, c, o_rd_data); fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
            end
        end

        // T2: Write all chunks for all experts
        for (c = 0; c < NUM_CHUNKS; c = c + 1) begin
            @(negedge clk);
            for (e = 0; e < NUM_EXPERTS; e = e + 1) begin
                i_wr_valid[e]     = 1;
                i_wr_chunk_idx[e] = c[CHUNK_IDX_W-1:0];
                i_wr_data[e]      = $urandom;
                ref_mem[e][c]     = i_wr_data[e];
            end
        end
        @(negedge clk);
        i_wr_valid = 0;
        @(posedge clk); #1;

        // Read back
        for (e = 0; e < NUM_EXPERTS; e = e + 1) begin
            for (c = 0; c < NUM_CHUNKS; c = c + 1) begin
                i_rd_expert    = e[EXPERT_IDX_W-1:0];
                i_rd_chunk_idx = c[CHUNK_IDX_W-1:0];
                #1;
                if (o_rd_data !== ref_mem[e][c]) begin
                    $display("FAIL T2: [%0d][%0d]=%h exp=%h", e, c, o_rd_data, ref_mem[e][c]);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
            end
        end

        // T3: Overwrite expert 1 only
        for (c = 0; c < NUM_CHUNKS; c = c + 1) begin
            ref_mem[1][c] = ~ref_mem[1][c];
        end
        for (c = 0; c < NUM_CHUNKS; c = c + 1) begin
            @(negedge clk);
            i_wr_valid    = 3'b010;
            i_wr_chunk_idx[0] = 0; i_wr_data[0] = 0;
            i_wr_chunk_idx[1] = c[CHUNK_IDX_W-1:0];
            i_wr_data[1]      = ref_mem[1][c];
            i_wr_chunk_idx[2] = 0; i_wr_data[2] = 0;
        end
        @(negedge clk);
        i_wr_valid = 0;
        @(posedge clk); #1;

        for (e = 0; e < NUM_EXPERTS; e = e + 1) begin
            for (c = 0; c < NUM_CHUNKS; c = c + 1) begin
                i_rd_expert    = e[EXPERT_IDX_W-1:0];
                i_rd_chunk_idx = c[CHUNK_IDX_W-1:0];
                #1;
                if (o_rd_data !== ref_mem[e][c]) begin
                    $display("FAIL T3: [%0d][%0d]=%h exp=%h", e, c, o_rd_data, ref_mem[e][c]);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
            end
        end

        // T4: Combinational check
        i_rd_expert = 0; i_rd_chunk_idx = 0; #1;
        if (o_rd_data !== ref_mem[0][0]) begin
            $display("FAIL T4a"); fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;
        i_rd_expert = 2; i_rd_chunk_idx = 63; #1;
        if (o_rd_data !== ref_mem[2][63]) begin
            $display("FAIL T4b"); fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
