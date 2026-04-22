// =============================================================================
// Testbench : tb_hamming_argmax
// Lang      : SystemVerilog
// DUT chain : centroid_bram -> hamming_sim -> argmax
// Coverage  : known XNOR+popcount values, correct class selection,
//             all five test vectors have hand-computed expected results.
// =============================================================================
`timescale 1ns/1ps

module tb_hamming_argmax;

    localparam int CLK_PERIOD = 10;
    localparam int D          = 8;
    localparam int C          = 4;
    localparam int ADDR_W     = 2;   // ceil(log2(4)) = 2
    localparam int SCORE_W    = 4;   // ceil(log2(8))+1 = 4
    localparam int ID_W       = 4;

    logic clk, rst_n;

    // ── centroid BRAM ports ───────────────────────────────────────────────────
    logic [ADDR_W-1:0] bram_rd_addr;
    logic              bram_rd_en;
    logic [D-1:0]      bram_rd_data;
    logic [ADDR_W-1:0] bram_wr_addr;
    logic              bram_wr_en;
    logic [D-1:0]      bram_wr_data;

    centroid_bram #(.D(D), .C(C), .ADDR_W(ADDR_W)) u_bram (
        .clk    (clk),
        .rd_addr(bram_rd_addr),
        .rd_en  (bram_rd_en),
        .rd_data(bram_rd_data),
        .wr_addr(bram_wr_addr),
        .wr_en  (bram_wr_en),
        .wr_data(bram_wr_data)
    );

    // ── hamming_sim ───────────────────────────────────────────────────────────
    logic [D-1:0]          hv_query;
    logic                  hv_valid;
    logic [C*SCORE_W-1:0]  score_out;
    logic                  scores_valid;
    logic                  ham_busy;

    hamming_sim #(.D(D), .C(C), .ADDR_W(ADDR_W), .SCORE_W(SCORE_W)) u_ham (
        .clk         (clk),
        .rst_n       (rst_n),
        .hv_query    (hv_query),
        .hv_valid    (hv_valid),
        .rd_addr     (bram_rd_addr),
        .rd_en       (bram_rd_en),
        .centroid    (bram_rd_data),
        .score_out   (score_out),
        .scores_valid(scores_valid),
        .busy        (ham_busy)
    );

    // ── argmax ────────────────────────────────────────────────────────────────
    logic [ID_W-1:0] class_id;
    logic            result_valid;

    argmax #(.C(C), .SCORE_W(SCORE_W), .ID_W(ID_W)) u_argmax (
        .clk         (clk),
        .rst_n       (rst_n),
        .scores_in   (score_out),
        .scores_valid(scores_valid),
        .class_id    (class_id),
        .result_valid(result_valid)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0, fail_cnt = 0;

    // Write one centroid into BRAM
    task automatic load_centroid(input logic [ADDR_W-1:0] idx,
                                  input logic [D-1:0]      vec);
        @(posedge clk); #1;
        bram_wr_addr = idx;
        bram_wr_data = vec;
        bram_wr_en   = 1'b1;
        @(posedge clk); #1;
        bram_wr_en   = 1'b0;
    endtask

    // Send one query and check the returned class
    task automatic run_inference(input logic [D-1:0]   query,
                                  input logic [ID_W-1:0] expected_class);
        @(posedge clk); #1;
        hv_query = query;
        hv_valid = 1'b1;
        @(posedge clk); #1;
        hv_valid = 1'b0;
        do @(posedge clk); while (!result_valid);
        if (class_id === expected_class) begin
            $display("[PASS] query=0x%02X -> class %0d (expected %0d)",
                     query, class_id, expected_class);
            pass_cnt++;
        end else begin
            $display("[FAIL] query=0x%02X -> class %0d (expected %0d)",
                     query, class_id, expected_class);
            fail_cnt++;
        end
    endtask

    initial begin
        rst_n        = 1'b0;
        hv_query     = '0;
        hv_valid     = 1'b0;
        bram_wr_en   = 1'b0;
        bram_wr_addr = '0;
        bram_wr_data = '0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Load centroids (D=8 bits each):
        //   class 0: 8'b0000_0000  (0x00)
        //   class 1: 8'b1111_0000  (0xF0)
        //   class 2: 8'b1111_1111  (0xFF)
        //   class 3: 8'b1010_1010  (0xAA)
        load_centroid(2'd0, 8'b0000_0000);
        load_centroid(2'd1, 8'b1111_0000);
        load_centroid(2'd2, 8'b1111_1111);
        load_centroid(2'd3, 8'b1010_1010);
        repeat(3) @(posedge clk);

        // ── Test 1: query = 0xFF -> best match class 2 (sim=8) ────────────────
        // XNOR(0xFF, 0xFF) = 0xFF -> popcount = 8  -> winner
        // XNOR(0xFF, 0xF0) = 0xF0 -> popcount = 4
        // XNOR(0xFF, 0x00) = 0x00 -> popcount = 0
        // XNOR(0xFF, 0xAA) = 0xAA -> popcount = 4
        $display("=== Test 1: query=0xFF -> class 2 ===");
        run_inference(8'hFF, 4'd2);

        // ── Test 2: query = 0x00 -> best match class 0 (sim=8) ────────────────
        $display("=== Test 2: query=0x00 -> class 0 ===");
        run_inference(8'h00, 4'd0);

        // ── Test 3: query = 0xF0 -> best match class 1 (sim=8) ────────────────
        $display("=== Test 3: query=0xF0 -> class 1 ===");
        run_inference(8'hF0, 4'd1);

        // ── Test 4: query = 0xAA -> best match class 3 (sim=8) ────────────────
        $display("=== Test 4: query=0xAA -> class 3 ===");
        run_inference(8'hAA, 4'd3);

        // ── Test 5: query = 0xE0 -> best match class 1 ────────────────────────
        // XNOR(0xE0, 0xF0) = ~(0x10) = 0xEF -> popcount = 7  -> winner
        // XNOR(0xE0, 0xFF) = ~(0x1F) = 0xE0 -> popcount = 3
        // XNOR(0xE0, 0x00) = ~(0xE0) = 0x1F -> popcount = 5
        // XNOR(0xE0, 0xAA) = ~(0x4A) = 0xB5 -> popcount = 5 (tie, class 0 lower idx wins)
        // So class 1 wins with 7
        $display("=== Test 5: query=0xE0 -> class 1 ===");
        run_inference(8'hE0, 4'd1);

        $display("\n=== hamming+argmax TB DONE: PASS=%0d FAIL=%0d ===",
                 pass_cnt, fail_cnt);
        $finish;
    end

    initial begin #(CLK_PERIOD * 3000); $display("[TIMEOUT]"); $finish; end
    initial begin $dumpfile("tb_hamming_argmax.vcd"); $dumpvars(0, tb_hamming_argmax); end

endmodule

