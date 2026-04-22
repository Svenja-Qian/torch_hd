`timescale 1ns / 1ps
module tb_expert_score_buffer;

    localparam int NE = 3, NC = 10, HW = 12;
    localparam int EW = 2, CW = 4;
    localparam int CLK_PERIOD = 10;

    logic clk, rst_n;
    logic            i_wr_en;
    logic [EW-1:0]   i_wr_expert;
    logic [CW-1:0]   i_wr_class;
    logic [HW-1:0]   i_wr_dist;
    logic [CW-1:0]   i_rd_class;
    logic [NE*HW-1:0] o_rd_dist_packed;

    expert_score_buffer #(.NUM_EXPERTS(NE),.NUM_CLASSES(NC),.HAM_WIDTH(HW)) dut(.*);

    initial clk=0;
    always #(CLK_PERIOD/2) clk=~clk;

    reg [HW-1:0] ref_s [0:NE-1][0:NC-1];
    integer pass_cnt, fail_cnt, e, c;
    reg [HW-1:0] got;

    initial begin
        $display("=== expert_score_buffer Testbench ===");
        pass_cnt=0; fail_cnt=0;
        rst_n=0; i_wr_en=0; i_wr_expert=0; i_wr_class=0; i_wr_dist=0; i_rd_class=0;
        repeat(3) @(posedge clk);
        rst_n=1;
        @(posedge clk); #1;

        // T1: Reset → 0
        for (c=0; c<NC; c=c+1) begin
            i_rd_class = c[CW-1:0]; #1;
            for (e=0; e<NE; e=e+1) begin
                got = o_rd_dist_packed[e*HW +: HW];
                if (got !== 0) begin
                    $display("FAIL T1: [%0d][%0d]=%0d", e, c, got); fail_cnt=fail_cnt+1;
                end else pass_cnt=pass_cnt+1;
            end
        end

        // T2: Write all entries
        for (e=0; e<NE; e=e+1) begin
            for (c=0; c<NC; c=c+1) begin
                ref_s[e][c] = (e*100 + c*7 + 1) & ((1<<HW)-1);
                @(negedge clk);
                i_wr_en     = 1;
                i_wr_expert = e[EW-1:0];
                i_wr_class  = c[CW-1:0];
                i_wr_dist   = ref_s[e][c];
            end
        end
        @(negedge clk); i_wr_en = 0;
        @(posedge clk); #1;

        for (c=0; c<NC; c=c+1) begin
            i_rd_class = c[CW-1:0]; #1;
            for (e=0; e<NE; e=e+1) begin
                got = o_rd_dist_packed[e*HW +: HW];
                if (got !== ref_s[e][c]) begin
                    $display("FAIL T2: [%0d][%0d]=%0d exp=%0d", e, c, got, ref_s[e][c]);
                    fail_cnt=fail_cnt+1;
                end else pass_cnt=pass_cnt+1;
            end
        end

        // T3: Overwrite one entry
        ref_s[1][5] = 999;
        @(negedge clk);
        i_wr_en=1; i_wr_expert=1; i_wr_class=5; i_wr_dist=999;
        @(negedge clk); i_wr_en=0;
        @(posedge clk); #1;

        i_rd_class = 5; #1;
        for (e=0; e<NE; e=e+1) begin
            got = o_rd_dist_packed[e*HW +: HW];
            if (got !== ref_s[e][5]) begin
                $display("FAIL T3: [%0d][5]=%0d exp=%0d", e, got, ref_s[e][5]);
                fail_cnt=fail_cnt+1;
            end else pass_cnt=pass_cnt+1;
        end
        i_rd_class = 4; #1;
        for (e=0; e<NE; e=e+1) begin
            got = o_rd_dist_packed[e*HW +: HW];
            if (got !== ref_s[e][4]) begin
                $display("FAIL T3b: [%0d][4]=%0d exp=%0d", e, got, ref_s[e][4]);
                fail_cnt=fail_cnt+1;
            end else pass_cnt=pass_cnt+1;
        end

        $display("=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt==0) $display("ALL TESTS PASSED");
        else $display("*** FAILURES DETECTED ***");
        $finish;
    end
endmodule
