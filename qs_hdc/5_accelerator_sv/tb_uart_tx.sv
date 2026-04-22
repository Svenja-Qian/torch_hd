// =============================================================================
// Testbench : tb_uart_tx  (with uart_rx loopback)
// Lang      : SystemVerilog
// =============================================================================
`timescale 1ns/1ps

module tb_uart_tx;

    localparam int CLK_PERIOD = 10;
    localparam int BAUD_DIV   = (100_000_000 / 115_200) - 1;
    localparam int BIT_NS     = 1_000_000_000 / 115_200;

    logic       clk, rst_n;
    logic [7:0] tx_data;
    logic       tx_valid;
    logic       tx, tx_busy;

    uart_tx #(.BAUD_DIV_W(16)) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .baud_div (16'(BAUD_DIV)),
        .tx_data  (tx_data),
        .tx_valid (tx_valid),
        .tx       (tx),
        .tx_busy  (tx_busy)
    );

    // loopback via uart_rx
    logic [7:0] lb_data;
    logic       lb_valid, lb_error;

    uart_rx #(.BAUD_DIV_W(16)) u_rx (
        .clk      (clk),
        .rst_n    (rst_n),
        .baud_div (16'(BAUD_DIV)),
        .rx       (tx),
        .rx_data  (lb_data),
        .rx_valid (lb_valid),
        .rx_error (lb_error)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int pass_cnt = 0, fail_cnt = 0;

    task automatic send_and_check(input logic [7:0] data);
        // wait until transmitter is free
        do @(posedge clk); while (tx_busy);
        @(posedge clk); #1;
        tx_data  = data;
        tx_valid = 1'b1;
        @(posedge clk); #1;
        tx_valid = 1'b0;
        // wait for loopback
        do @(posedge clk); while (!lb_valid);
        if (lb_data === data) begin
            $display("[PASS] Loopback 0x%02X OK", data); pass_cnt++;
        end else begin
            $display("[FAIL] Sent 0x%02X, received 0x%02X", data, lb_data); fail_cnt++;
        end
    endtask

    initial begin
        rst_n    = 1'b0;
        tx_data  = '0;
        tx_valid = 1'b0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        $display("=== Test 1: 0xA5 loopback ==="); send_and_check(8'hA5);
        $display("=== Test 2: 0x00 loopback ==="); send_and_check(8'h00);
        $display("=== Test 3: 0xFF loopback ==="); send_and_check(8'hFF);

        $display("=== Test 4: tx_busy flag ===");
        do @(posedge clk); while (tx_busy); #1;
        tx_data  = 8'hBB; tx_valid = 1'b1;
        @(posedge clk); #1; tx_valid = 1'b0;
        @(posedge clk);
        if (tx_busy) begin
            $display("[PASS] tx_busy asserted during TX"); pass_cnt++;
        end else begin
            $display("[FAIL] tx_busy not asserted"); fail_cnt++;
        end
        do @(posedge clk); while (!lb_valid);

        $display("\n=== uart_tx TB DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        $finish;
    end

    initial begin #(BIT_NS * 300); $display("[TIMEOUT]"); $finish; end
    initial begin $dumpfile("tb_uart_tx.vcd"); $dumpvars(0, tb_uart_tx); end

endmodule
