// =============================================================================
// Module : centroid_bram
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : Dual-port BRAM: C centroids × D bits.
//          Port A: synchronous read  (1-cycle latency).
//          Port B: synchronous write (pre-load only; tie wr_en=0 at inference).
//
// Centroid file format (for $readmemh):
//   One line per class, D/4 hex characters per line (D bits, MSB-first).
//   Example for D=8, C=3:
//     FF       ← class 0 centroid = 8'b11111111
//     AA       ← class 1 centroid = 8'b10101010
//     0F       ← class 2 centroid = 8'b00001111
// =============================================================================
`timescale 1ns/1ps

module centroid_bram #(
    parameter int    D         = 4096,
    parameter int    C         = 6,
    parameter int    ADDR_W    = 3,          // ceil(log2(C))
    parameter        INIT_FILE = "",         // path to hex init; "" = skip
    parameter bit    REQUIRE_INIT = 1'b0
)(
    input  logic              clk,
    // read port (inference)
    input  logic [ADDR_W-1:0] rd_addr,
    input  logic              rd_en,
    output logic [D-1:0]      rd_data,
    // write port (pre-load)
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic              wr_en,
    input  logic [D-1:0]      wr_data
);

    (* ram_style = "block" *)
    logic [D-1:0] mem [0:C-1];

    // Initialise: load from file if provided, otherwise zero-fill.
    initial begin
        rd_data = '0;
        for (int i = 0; i < C; i++)
            mem[i] = '0;
        if (INIT_FILE != "") begin
`ifndef SYNTHESIS
            int init_fd;
            init_fd = $fopen(INIT_FILE, "r");
            if (init_fd == 0) begin
                $fatal(1,
                       "centroid_bram could not open INIT_FILE='%s'. In Vivado/XSim, add the .mem file as a Simulation Source or copy it into the simulation run directory.",
                       INIT_FILE);
            end
            $fclose(init_fd);
`endif
            $readmemh(INIT_FILE, mem, 0, C-1);
        end else if (REQUIRE_INIT) begin
`ifndef SYNTHESIS
            $fatal(1, "centroid_bram INIT_FILE is empty but REQUIRE_INIT=1");
`endif
        end
    end

    // Synchronous write
    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // Synchronous read (1-cycle latency)
    always_ff @(posedge clk) begin
        if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule
