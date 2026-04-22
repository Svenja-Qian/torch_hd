// =============================================================================
// Testbench : tb_frame_ctrl
// Lang      : SystemVerilog
// Uses FEAT_NUM=4 for fast simulation.
// =============================================================================
`timescale 1ns/1ps

module tb_frame_ctrl;

    localparam int CLK_PERIOD = 10;
    localparam int FEAT_NUM   = 4;
    localparam int DATA_W     = 16;

    logic      clk, rst_n;
    logic [7:0] rx_data;
    logic       rx_valid;
    logic [FEAT_NUM*DATA_W-1:0] feature_vec;
    logic                       frame_valid;
    logic [10:0]                byte_count;
    logic                       frame_ready;

    frame_ctrl #(.FEAT_NUM(FEAT_NUM), .BYTES_PER_FEAT(2), .DATA_W(DATA_W)) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .frame_ready(frame_ready),
        .rx_data    (rx_data),
        .rx_valid   (rx_valid),
        .feature_vec(feature_vec),
        .frame_valid(frame_valid),
        .byte_count (byte_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    task automatic push_byte(input logic [7:0] d);
        @(posedge clk); #1;
        rx_data  = d;
        rx_valid = 1'b1;
        @(posedge clk); #1;
        rx_valid = 1'b0;
        repeat(2) @(posedge clk);
    endtask

    // Send a valid frame: features 0x0001, 0x0002, 0x0003, 0x0004
    task automatic send_valid_frame;
        push_byte(8'hAA);
        push_byte(8'h00); push_byte(8'h01);
        push_byte(8'h00); push_byte(8'h02);
        push_byte(8'h00); push_byte(8'h03);
        push_byte(8'h00); push_byte(8'h04);
        push_byte(8'h55);
    endtask

    int pass_cnt = 0, fail_cnt = 0;

    initial begin
        rst_n    = 1'b0;
        rx_data  = '0;
        rx_valid = 1'b0;
        frame_ready = 1'b1;
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);

        // Test 1: valid frame, check feature ordering
        // feature[0] → MSB slot, feature[3] → LSB slot
        $display("=== Test 1: Valid frame ===");
        fork
            send_valid_frame;
            begin
                do @(posedge clk); while (!frame_valid);
                begin
                    logic [15:0] f0, f1, f2, f3;
                    f0 = feature_vec[FEAT_NUM*DATA_W-1       -: 16];
                    f1 = feature_vec[FEAT_NUM*DATA_W-1-16    -: 16];
                    f2 = feature_vec[FEAT_NUM*DATA_W-1-32    -: 16];
                    f3 = feature_vec[FEAT_NUM*DATA_W-1-48    -: 16];
                    if (f0 === 16'h0001 && f1 === 16'h0002 &&
                        f2 === 16'h0003 && f3 === 16'h0004) begin
                        $display("[PASS] Features assembled in correct order"); pass_cnt++;
                    end else begin
                        $display("[FAIL] f0=%04X f1=%04X f2=%04X f3=%04X", f0,f1,f2,f3);
                        fail_cnt++;
                    end
                end
            end
        join

        // Test 2: no frame_valid without SOF
        $display("=== Test 2: Missing SOF ===");
        push_byte(8'h00); push_byte(8'h01);
        push_byte(8'h00); push_byte(8'h02);
        push_byte(8'h55);
        repeat(10) @(posedge clk);
        if (!frame_valid) begin
            $display("[PASS] No spurious frame_valid"); pass_cnt++;
        end else begin
            $display("[FAIL] Spurious frame_valid without SOF"); fail_cnt++;
        end

        // Test 3: second consecutive frame
        $display("=== Test 3: Consecutive frame ===");
        fork
            send_valid_frame;
            begin
                do @(posedge clk); while (!frame_valid);
                $display("[PASS] Second frame received OK"); pass_cnt++;
            end
        join

        $display("\n=== frame_ctrl TB DONE: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
        $finish;
    end

    initial begin #(CLK_PERIOD*2000); $display("[TIMEOUT]"); $finish; end
    initial begin $dumpfile("tb_frame_ctrl.vcd"); $dumpvars(0, tb_frame_ctrl); end

endmodule
