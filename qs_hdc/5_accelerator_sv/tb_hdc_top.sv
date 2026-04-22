`timescale 1ns/1ps

module tb_hdc_top;

    localparam int CLK_PERIOD = 10;
    localparam int BAUD_RATE  = 115_200;
    localparam int BAUD_DIV   = (100_000_000 / BAUD_RATE) - 1;
    localparam int BIT_NS     = 1_000_000_000 / BAUD_RATE;

    localparam int D        = 16;
    localparam int C        = 6;
    localparam int FEAT_NUM = 4;
    localparam int ACC_W    = 26;
    localparam int SCORE_W  = 5;
    localparam int ID_W     = 4;
    localparam int ADDR_W   = 3;
    localparam int FRAME_BYTES        = 10; // SOF + 8 data bytes + EOF
    localparam int UART_BITS_PER_BYTE = 10; // 8N1
    localparam int MIN_FIRST_RESULT_NS =
        (FRAME_BYTES * UART_BITS_PER_BYTE * BIT_NS) +
        (D * FEAT_NUM * CLK_PERIOD) +
        CLK_PERIOD +
        ((C * 4 + 2) * CLK_PERIOD) +
        (UART_BITS_PER_BYTE * BIT_NS);

    logic clk;
    logic rst_n;
    logic rx;
    logic tx;

    logic [7:0] result_byte;
    int pass_cnt = 0;
    int fail_cnt = 0;

    hdc_top #(
        .D                 (D),
        .C                 (C),
        .FEAT_NUM          (FEAT_NUM),
        .DATA_W            (16),
        .ACC_W             (ACC_W),
        .SCORE_W           (SCORE_W),
        .ID_W              (ID_W),
        .ADDR_W            (ADDR_W),
        .BAUD_DIV_W        (16),
        .BAUD_DIV          (BAUD_DIV),
        // Vivado/XSim resolves this relative to the simulation run directory.
        // Add centroids_tb.mem as a Simulation Source, or copy it beside xsim.dir.
        .CENTROID_INIT_FILE("centroids_tb.mem"),
        .WEIGHT_INIT_FILE  ("weights_tb.mem"),
        .ALLOW_DUMMY_WEIGHTS(1'b0)
    ) dut (
        .clk  (clk),
        .rst_n(rst_n),
        .rx   (rx),
        .tx   (tx)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    task automatic uart_send_byte(input logic [7:0] d);
        rx = 1'b0;
        #(BIT_NS);
        for (int i = 0; i < 8; i++) begin
            rx = d[i];
            #(BIT_NS);
        end
        rx = 1'b1;
        #(BIT_NS);
    endtask

    task automatic send_frame4(
        input logic [15:0] f0,
        input logic [15:0] f1,
        input logic [15:0] f2,
        input logic [15:0] f3
    );
        uart_send_byte(8'hAA);
        uart_send_byte(f0[15:8]); uart_send_byte(f0[7:0]);
        uart_send_byte(f1[15:8]); uart_send_byte(f1[7:0]);
        uart_send_byte(f2[15:8]); uart_send_byte(f2[7:0]);
        uart_send_byte(f3[15:8]); uart_send_byte(f3[7:0]);
        uart_send_byte(8'h55);
    endtask

    task automatic uart_recv_byte(output logic [7:0] data);
        @(negedge tx);
        #(BIT_NS + BIT_NS/2);
        for (int i = 0; i < 8; i++) begin
            data[i] = tx;
            if (i < 7)
                #(BIT_NS);
        end
        #(BIT_NS/2 + BIT_NS);
    endtask

    task automatic golden_infer(
        input logic [15:0] f0,
        input logic [15:0] f1,
        input logic [15:0] f2,
        input logic [15:0] f3,
        input logic [3:0]  expected_class,
        input string       test_name
    );
        $display("=== %s ===", test_name);
        $display("    Sending feats: %04X %04X %04X %04X  expecting class %0d",
                 f0, f1, f2, f3, expected_class);

        fork
            send_frame4(f0, f1, f2, f3);
            uart_recv_byte(result_byte);
        join

        $display("    Received byte: 0x%02X  (class_id=%0d)", result_byte, result_byte[3:0]);

        if (result_byte[7:4] !== 4'h0) begin
            $display("[FAIL] Bad format: high nibble = 0x%X", result_byte[7:4]);
            fail_cnt++;
        end else if (result_byte[3:0] === expected_class) begin
            $display("[PASS] class_id = %0d (correct)", result_byte[3:0]);
            pass_cnt++;
        end else begin
            $display("[FAIL] Expected class %0d, got class %0d", expected_class, result_byte[3:0]);
            fail_cnt++;
        end
    endtask

    initial begin
        rx    = 1'b1;
        rst_n = 1'b0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        $display("[INFO] tb_hdc_top first classification result will not appear before about %0d ns", MIN_FIRST_RESULT_NS);
        $display("[INFO] In XSim, use Run All or run for at least 1000000 ns");
        #(BIT_NS);

        golden_infer(16'h0001, 16'h7FFF, 16'h7FFF, 16'h0001, 4'd0, "Test A: hv=0x9669 -> class 0");
        #(BIT_NS * 3);

        golden_infer(16'h7FFF, 16'h0001, 16'h0001, 16'h7FFF, 4'd1, "Test B: hv=0x6996 -> class 1");
        #(BIT_NS * 3);

        golden_infer(16'hFFFF, 16'hFFFF, 16'hFFFF, 16'hFFFF, 4'd5, "Test C: hv=0x0000 -> class 5");
        #(BIT_NS * 3);

        golden_infer(16'h0001, 16'h7FFF, 16'h7FFF, 16'h0001, 4'd0, "Test D: re-arm -> class 0");
        #(BIT_NS * 3);

        $display("=== Test E: Missing EOF -> no spurious TX output ===");
        uart_send_byte(8'hAA);
        for (int i = 0; i < FEAT_NUM; i++) begin
            uart_send_byte(8'h00);
            uart_send_byte(8'h01);
        end
        #(BIT_NS * 30);
        if (tx === 1'b1) begin
            $display("[PASS] TX remains idle -> no spurious output");
            pass_cnt++;
        end else begin
            $display("[FAIL] TX went low unexpectedly for incomplete frame");
            fail_cnt++;
        end

        $display("\n=== tb_hdc_top DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("    ALL TESTS PASSED");
        else
            $display("    *** %0d TEST(S) FAILED ***", fail_cnt);

        $finish;
    end

    initial begin
        #(BIT_NS * 25_000);
        $display("[TIMEOUT] tb_hdc_top exceeded maximum simulation time");
        $finish;
    end

    initial begin
        $dumpfile("tb_hdc_top.vcd");
        $dumpvars(0, tb_hdc_top);
    end

endmodule
