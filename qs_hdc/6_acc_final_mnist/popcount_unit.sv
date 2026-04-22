//==============================================================================
// popcount_unit.sv
// Pure combinational popcount — counts the number of '1' bits in i_data.
// Simple iterative implementation (synthesizes to adder tree).
//==============================================================================

module popcount_unit #(
    parameter int unsigned WIDTH   = 32,
    parameter int unsigned COUNT_W = $clog2(WIDTH + 1)   // 6 for WIDTH=32
) (
    input  logic [WIDTH-1:0]    i_data,
    output logic [COUNT_W-1:0]  o_count
);

    always_comb begin
        o_count = '0;
        for (int i = 0; i < WIDTH; i++) begin
            o_count = o_count + {{(COUNT_W-1){1'b0}}, i_data[i]};
        end
    end

endmodule
