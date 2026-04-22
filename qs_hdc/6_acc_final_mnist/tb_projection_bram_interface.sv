//==============================================================================
// tb_projection_bram_interface.sv
//==============================================================================
`timescale 1ns / 1ps
module tb_projection_bram_interface;

    localparam int NC = 4;          // small NUM_CHUNKS for testing
    localparam int BAW = 10;        // BRAM_ADDR_W
    localparam int FIW = 8;         // FEAT_IDX_W
    localparam int FW = 16;         // FEAT_WIDTH
    localparam int CW = 32;         // CHUNK_WIDTH
    localparam int CIW = $clog2(NC);
    localparam int CLK = 10;

    logic clk, rst_n;
    logic i_enable, i_feat_valid;
    logic [FIW-1:0] i_feat_index;
    logic signed [FW-1:0] i_feat_value;
    logic o_bram_en;
    logic [BAW-1:0] o_bram_addr;
    logic [CW-1:0] i_bram_rdata;
    logic o_chunk_valid;
    logic [CW-1:0] o_chunk_data;
    logic [CIW-1:0] o_chunk_idx;
    logic signed [FW-1:0] o_feat_value_out;
    logic o_feat_done, o_busy;

    projection_bram_interface #(
        .NUM_CHUNKS(NC), .BRAM_ADDR_W(BAW), .FEAT_IDX_W(FIW),
        .FEAT_WIDTH(FW), .CHUNK_WIDTH(CW)
    ) dut (.*);

    initial clk = 0;
    always #(CLK/2) clk = ~clk;

    // Simple BRAM model: returns address as data (for easy checking)
    always_ff @(posedge clk) begin
        if (o_bram_en)
            i_bram_rdata <= {22'd0, o_bram_addr};
        else
            i_bram_rdata <= '0;
    end

    integer pass_cnt, fail_cnt;
    integer i, expected_addr;

    initial begin
        $display("=== projection_bram_interface Testbench ===");
        pass_cnt = 0; fail_cnt = 0;
        rst_n = 0; i_enable = 0; i_feat_valid = 0;
        i_feat_index = 0; i_feat_value = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // T1: Enable and trigger feature index=2, value=42
        i_enable = 1;
        @(negedge clk);
        i_feat_valid = 1;
        i_feat_index = 2;
        i_feat_value = 16'sd42;
        @(negedge clk);
        i_feat_valid = 0;

        // Wait for all chunks to come out (NC chunks + 1 pipeline delay)
        for (i = 0; i < NC + 3; i = i + 1) begin
            @(posedge clk); #1;
            if (o_chunk_valid) begin
                // Check chunk_idx
                expected_addr = 2 * NC + o_chunk_idx;
                if (o_chunk_data[BAW-1:0] !== expected_addr[BAW-1:0]) begin
                    $display("FAIL T1: chunk_idx=%0d data=%0d exp_addr=%0d",
                             o_chunk_idx, o_chunk_data[BAW-1:0], expected_addr);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;

                // Check feat_value alignment
                if (o_feat_value_out !== 16'sd42) begin
                    $display("FAIL T1: feat_val=%0d exp=42", o_feat_value_out);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;

                // Check feat_done on last chunk
                if (o_chunk_idx == NC - 1) begin
                    if (o_feat_done !== 1) begin
                        $display("FAIL T1: feat_done not set on last chunk");
                        fail_cnt = fail_cnt + 1;
                    end else pass_cnt = pass_cnt + 1;
                end
            end
        end

        // T2: Check busy goes low
        @(posedge clk); #1;
        if (o_busy !== 0) begin
            $display("FAIL T2: still busy"); fail_cnt = fail_cnt + 1;
        end else pass_cnt = pass_cnt + 1;

        // T3: Feature index=0, value=-10
        @(negedge clk);
        i_feat_valid = 1;
        i_feat_index = 0;
        i_feat_value = -16'sd10;
        @(negedge clk);
        i_feat_valid = 0;

        for (i = 0; i < NC + 3; i = i + 1) begin
            @(posedge clk); #1;
            if (o_chunk_valid) begin
                expected_addr = 0 * NC + o_chunk_idx;
                if (o_chunk_data[BAW-1:0] !== expected_addr[BAW-1:0]) begin
                    $display("FAIL T3: chunk=%0d addr=%0d exp=%0d",
                             o_chunk_idx, o_chunk_data[BAW-1:0], expected_addr);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;

                if (o_feat_value_out !== -16'sd10) begin
                    $display("FAIL T3: feat_val=%0d", o_feat_value_out);
                    fail_cnt = fail_cnt + 1;
                end else pass_cnt = pass_cnt + 1;
            end
        end

        // T4: Disabled expert should not generate BRAM reads
        i_enable = 0;
        @(negedge clk);
        i_feat_valid = 1;
        i_feat_index = 5;
        i_feat_value = 16'sd99;
        @(negedge clk);
        i_feat_valid = 0;
        repeat (NC + 3) begin
            @(posedge clk); #1;
            if (o_bram_en) begin
                $display("FAIL T4: BRAM read when disabled");
                fail_cnt = fail_cnt + 1;
            end
        end
        pass_cnt = pass_cnt + 1; // if we get here, no spurious reads

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
