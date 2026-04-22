//==============================================================================
// input_buffer.sv
// Small synchronous FIFO decoupling external feature input from internal
// encoding pipeline.  Shift-register based (no BRAM).
//
// Write side: valid/ready handshake
// Read side:  rd_en / empty protocol
//==============================================================================

module input_buffer #(
    parameter int unsigned FEAT_WIDTH   = 16,
    parameter int unsigned FEAT_IDX_W   = 8,
    parameter int unsigned DEPTH        = 8,
    // Derived
    parameter int unsigned DATA_W       = FEAT_WIDTH + FEAT_IDX_W + 1, // +1 for feat_last
    parameter int unsigned PTR_W        = $clog2(DEPTH + 1)
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // Write side (external)
    input  logic                        i_wr_valid,
    output logic                        o_wr_ready,
    input  logic signed [FEAT_WIDTH-1:0] i_feat_value,
    input  logic [FEAT_IDX_W-1:0]       i_feat_index,
    input  logic                        i_feat_last,

    // Read side (internal)
    input  logic                        i_rd_en,
    output logic                        o_empty,
    output logic                        o_full,
    output logic signed [FEAT_WIDTH-1:0] o_feat_value,
    output logic [FEAT_IDX_W-1:0]       o_feat_index,
    output logic                        o_feat_last
);

    // -------------------------------------------------------------------------
    // Storage
    // -------------------------------------------------------------------------
    logic [DATA_W-1:0] mem [DEPTH];
    logic [PTR_W-1:0]  wr_ptr, rd_ptr, count;

    // -------------------------------------------------------------------------
    // Pack / Unpack
    // -------------------------------------------------------------------------
    logic [DATA_W-1:0] wr_data;
    logic [DATA_W-1:0] rd_data;

    assign wr_data = {i_feat_last, i_feat_index, i_feat_value};
    assign rd_data = mem[rd_ptr[$clog2(DEPTH)-1:0]];

    assign o_feat_value = rd_data[FEAT_WIDTH-1:0];
    assign o_feat_index = rd_data[FEAT_WIDTH +: FEAT_IDX_W];
    assign o_feat_last  = rd_data[DATA_W-1];

    // -------------------------------------------------------------------------
    // Flags
    // -------------------------------------------------------------------------
    assign o_empty    = (count == '0);
    assign o_full     = (count == PTR_W'(DEPTH));
    assign o_wr_ready = !o_full;

    // -------------------------------------------------------------------------
    // Pointer and count logic
    // -------------------------------------------------------------------------
    wire do_write = i_wr_valid && o_wr_ready;
    wire do_read  = i_rd_en && !o_empty;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            case ({do_write, do_read})
                2'b10: begin // write only
                    mem[wr_ptr[$clog2(DEPTH)-1:0]] <= wr_data;
                    wr_ptr <= wr_ptr + 1'b1;
                    count  <= count + 1'b1;
                end
                2'b01: begin // read only
                    rd_ptr <= rd_ptr + 1'b1;
                    count  <= count - 1'b1;
                end
                2'b11: begin // simultaneous read + write
                    mem[wr_ptr[$clog2(DEPTH)-1:0]] <= wr_data;
                    wr_ptr <= wr_ptr + 1'b1;
                    rd_ptr <= rd_ptr + 1'b1;
                    // count stays the same
                end
                default: ; // no operation
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Pointer wrap: use modular arithmetic on $clog2(DEPTH) bits
    // -------------------------------------------------------------------------
    // Note: wr_ptr and rd_ptr are PTR_W bits wide to hold [0, DEPTH].
    // The memory index uses only the lower $clog2(DEPTH) bits.
    // This works correctly for power-of-2 DEPTH. For non-power-of-2,
    // additional wrap logic would be needed.

endmodule
