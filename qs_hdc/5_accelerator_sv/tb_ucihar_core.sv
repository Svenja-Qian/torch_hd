`timescale 1ns/1ps

module tb_ucihar_core #(
    parameter int D        = 1000,
    parameter int C        = 6,
    parameter int FEAT_NUM = 561,
    parameter int DATA_W   = 16,
    parameter int ACC_W    = 26,
    parameter int SCORE_W  = 13,
    parameter int ID_W     = 4,
    parameter int ADDR_W   = 3,
    parameter int NUM_CASES = 5,
    parameter string WEIGHT_INIT_FILE   = "weights_D1000_seed123.memb",
    parameter string CENTROID_INIT_FILE = "centroids_D1000_seed123.memh",
    parameter string TESTCASE_FILE      = "testcases_D1000_seed123_n5_off0.memh"
)();

    localparam int CLK_PERIOD = 10;
    localparam int CASE_STRIDE = FEAT_NUM + 1;
    localparam int TOTAL_WORDS = NUM_CASES * CASE_STRIDE;

    logic clk;
    logic rst_n;

    logic [15:0] tc_mem [0:TOTAL_WORDS-1];

    logic [FEAT_NUM*DATA_W-1:0] feature_vec;
    logic frame_valid;

    logic [D*ACC_W-1:0] acc_out;
    logic proj_valid;
    logic proj_busy;

    logic [D-1:0] hv_out;
    logic hv_valid;

    logic [ADDR_W-1:0] bram_rd_addr;
    logic bram_rd_en;
    logic [D-1:0] bram_rd_data;

    logic [C*SCORE_W-1:0] score_out;
    logic scores_valid;
    logic ham_busy;

    logic [ID_W-1:0] class_id;
    logic result_valid;

    centroid_bram #(
        .D(D),
        .C(C),
        .ADDR_W(ADDR_W),
        .INIT_FILE(CENTROID_INIT_FILE),
        .REQUIRE_INIT(1'b1)
    ) u_bram (
        .clk    (clk),
        .rd_addr(bram_rd_addr),
        .rd_en  (bram_rd_en),
        .rd_data(bram_rd_data),
        .wr_addr('0),
        .wr_en  (1'b0),
        .wr_data('0)
    );

    projection_engine #(
        .D(D),
        .FEAT_NUM(FEAT_NUM),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .WEIGHT_INIT_FILE(WEIGHT_INIT_FILE),
        .ALLOW_DUMMY_WEIGHTS(1'b0)
    ) u_proj (
        .clk        (clk),
        .rst_n      (rst_n),
        .feature_vec(feature_vec),
        .frame_valid(frame_valid),
        .acc_out    (acc_out),
        .proj_valid (proj_valid),
        .busy       (proj_busy)
    );

    binarizer #(
        .D(D),
        .ACC_W(ACC_W)
    ) u_bin (
        .clk       (clk),
        .rst_n     (rst_n),
        .acc_in    (acc_out),
        .proj_valid(proj_valid),
        .hv_out    (hv_out),
        .hv_valid  (hv_valid)
    );

    hamming_sim #(
        .D(D),
        .C(C),
        .ADDR_W(ADDR_W),
        .SCORE_W(SCORE_W)
    ) u_ham (
        .clk         (clk),
        .rst_n       (rst_n),
        .hv_query    (hv_out),
        .hv_valid    (hv_valid),
        .rd_addr     (bram_rd_addr),
        .rd_en       (bram_rd_en),
        .centroid    (bram_rd_data),
        .score_out   (score_out),
        .scores_valid(scores_valid),
        .busy        (ham_busy)
    );

    argmax #(
        .C(C),
        .SCORE_W(SCORE_W),
        .ID_W(ID_W)
    ) u_argmax (
        .clk         (clk),
        .rst_n       (rst_n),
        .scores_in   (score_out),
        .scores_valid(scores_valid),
        .class_id    (class_id),
        .result_valid(result_valid)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    task automatic drive_case(input int case_idx);
        logic [15:0] exp;
        begin
            exp = tc_mem[case_idx * CASE_STRIDE];
            feature_vec = '0;
            for (int k = 0; k < FEAT_NUM; k++) begin
                feature_vec[(FEAT_NUM-1-k)*DATA_W +: DATA_W] = tc_mem[case_idx * CASE_STRIDE + 1 + k];
            end

            @(posedge clk); #1;
            frame_valid = 1'b1;
            @(posedge clk); #1;
            frame_valid = 1'b0;

            do @(posedge clk); while (!result_valid);
            if (class_id === exp[ID_W-1:0]) begin
                pass_cnt++;
            end else begin
                fail_cnt++;
                $display("[FAIL] case %0d expected=%0d got=%0d", case_idx, exp[ID_W-1:0], class_id);
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        frame_valid = 1'b0;
        feature_vec = '0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        $readmemh(TESTCASE_FILE, tc_mem);

        for (int i = 0; i < NUM_CASES; i++) begin
            drive_case(i);
            repeat(5) @(posedge clk);
        end

        $display("=== tb_ucihar_core DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 2000000);
        $display("[TIMEOUT] tb_ucihar_core exceeded maximum simulation time");
        $finish;
    end

    initial begin
        $dumpfile("tb_ucihar_core.vcd");
        $dumpvars(0, tb_ucihar_core);
    end

endmodule
