// =============================================================================
// Testbench : tb_uart_rx
// Lang      : SystemVerilog
// =============================================================================
`timescale 1ns/1ps

module tb_uart_rx;

    localparam int CLK_PERIOD = 10;
    localparam int BAUD_RATE  = 115_200;
    localparam int BAUD_DIV   = (100_000_000 / BAUD_RATE) - 1;
    localparam int BIT_NS     = 1_000_000_000 / BAUD_RATE;

    logic       clk, rst_n, rx;
    logic [7:0] rx_data;
    logic       rx_valid, rx_error;

    uart_rx #(.BAUD_DIV_W(16)) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .baud_div (16'(BAUD_DIV)),
        .rx       (rx),
        .rx_data  (rx_data),
        .rx_valid (rx_valid),
        .rx_error (rx_error)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── send one 8N1 byte on RX ───────────────────────────────────────────────
    task automatic send_byte(input logic [7:0] data);
        rx = 1'b0;           // start bit
        #(BIT_NS);
        for (int i = 0; i < 8; i++) begin
            rx = data[i];    // LSB first
            #(BIT_NS);
        end
        rx = 1'b1;           // stop bit
        #(BIT_NS);
    endtask

    task automatic send_framing_error(input logic [7:0] data);
        rx = 1'b0; #(BIT_NS);
        for (int i = 0; i < 8; i++) begin rx = data[i]; #(BIT_NS); end
        rx = 1'b0; #(BIT_NS);  // bad stop bit
        rx = 1'b1; #(BIT_NS);
    endtask

    int pass_cnt = 0, fail_cnt = 0;

    task automatic check_byte(input logic [7:0] expected);
        do @(posedge clk); while (!rx_valid);
        if (rx_data === expected) begin
            $display("[PASS] Received 0x%02X", rx_data);
            pass_cnt++;
        end else begin
            $display("[FAIL] Expected 0x%02X, Got 0x%02X", expected, rx_data);
            fail_cnt++;
        end
    endtask

    initial begin
        rx    = 1'b1;
        rst_n = 1'b0;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        #100;

        $display("\n=== Test 1: Single byte 0x55 ===");
        fork send_byte(8'h55);  check_byte(8'h55); join

        $display("=== Test 2: Single byte 0xAA ===");
        fork send_byte(8'hAA);  check_byte(8'hAA); join

        $display("=== Test 3: Back-to-back 0x01 0x02 0x03 ===");
        fork
            begin send_byte(8'h01); send_byte(8'h02); send_byte(8'h03); end
            begin check_byte(8'h01); check_byte(8'h02); check_byte(8'h03); end
        join

        $display("=== Test 4: Framing error ===");
        begin
            int err_seen;
            err_seen = 0;
            fork
                send_framing_error(8'hFF);
                begin
                    repeat((BIT_NS/CLK_PERIOD)*12 + 20) begin
                        @(posedge clk);
                        #1;
                        if (rx_error)
                            err_seen = 1;
                    end
                end
            join
            if (err_seen) begin
                $display("[PASS] rx_error asserted"); pass_cnt++;
            end else begin
                $display("[FAIL] rx_error NOT asserted"); fail_cnt++;
            end
        end

        $display("=== Test 5: Byte 0x00 ===");
        fork send_byte(8'h00);  check_byte(8'h00); join

        #(BIT_NS * 2);
        $display("\n=== uart_rx TB DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        $finish;
    end

    initial begin
        #(BIT_NS * 300);
        $display("[TIMEOUT]"); $finish;
    end

    initial begin $dumpfile("tb_uart_rx.vcd"); $dumpvars(0, tb_uart_rx); end

endmodule
