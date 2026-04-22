//==============================================================================
// accumulator_bank.sv
// Maintains HV_DIM signed accumulators. Supports clear, accumulate, binarize.
// Accumulate: per bit of proj_chunk, add or subtract feat_value.
// Binarize: output chunk-wise binary HV (acc>=0 → 1, acc<0 → 0).
//==============================================================================

module accumulator_bank #(
    parameter int unsigned HV_DIM      = 2048,
    parameter int unsigned CHUNK_WIDTH  = 32,
    parameter int unsigned NUM_CHUNKS   = HV_DIM / CHUNK_WIDTH,  // 64
    parameter int unsigned ACC_WIDTH    = 24,
    parameter int unsigned FEAT_WIDTH   = 16,
    parameter int unsigned CHUNK_IDX_W  = $clog2(NUM_CHUNKS)     // 6
) (
    input  logic                            clk,
    input  logic                            rst_n,

    // Control
    input  logic                            i_clear,            // pulse: clear all accumulators
    output logic                            o_acc_ready,        // can accept accumulate data

    // Accumulate interface (from projection_bram_interface)
    input  logic                            i_acc_valid,
    input  logic signed [FEAT_WIDTH-1:0]    i_feat_value,
    input  logic [CHUNK_WIDTH-1:0]          i_proj_chunk,
    input  logic [CHUNK_IDX_W-1:0]          i_chunk_idx,

    // Binarize control
    input  logic                            i_binarize_start,   // pulse
    output logic                            o_binarize_done,

    // Binarize output (to query_hv_buffer, chunk-wise)
    output logic                            o_qhv_valid,
    output logic [CHUNK_IDX_W-1:0]          o_qhv_chunk_idx,
    output logic [CHUNK_WIDTH-1:0]          o_qhv_chunk_data
);

    // -------------------------------------------------------------------------
    // Accumulator storage: NUM_CHUNKS × CHUNK_WIDTH accumulators
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc [NUM_CHUNKS][CHUNK_WIDTH];

    // -------------------------------------------------------------------------
    // FSM: IDLE, CLEARING, ACCUMULATING (implicit), BINARIZING
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE      = 2'd0,
        S_CLEARING  = 2'd1,
        S_BINARIZE  = 2'd2
    } state_t;

    state_t state_q;
    logic [CHUNK_IDX_W-1:0] op_chunk_cnt;  // chunk counter for clear/binarize

    // -------------------------------------------------------------------------
    // Ready signal
    // -------------------------------------------------------------------------
    assign o_acc_ready = (state_q == S_IDLE);

    // -------------------------------------------------------------------------
    // Main logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q         <= S_CLEARING;
            op_chunk_cnt    <= '0;
            o_binarize_done <= 1'b0;
            o_qhv_valid     <= 1'b0;
            o_qhv_chunk_idx <= '0;
            o_qhv_chunk_data<= '0;
        end else begin
            // Defaults
            o_qhv_valid     <= 1'b0;
            o_binarize_done <= 1'b0;

            case (state_q)
                S_IDLE: begin
                    if (i_clear) begin
                        state_q      <= S_CLEARING;
                        op_chunk_cnt <= '0;
                    end else if (i_binarize_start) begin
                        state_q      <= S_BINARIZE;
                        op_chunk_cnt <= '0;
                    end else if (i_acc_valid) begin
                        // Accumulate: for each bit in proj_chunk, add or subtract
                        for (int b = 0; b < CHUNK_WIDTH; b++) begin
                            if (i_proj_chunk[b])
                                acc[i_chunk_idx][b] <= acc[i_chunk_idx][b] + i_feat_value;
                            else
                                acc[i_chunk_idx][b] <= acc[i_chunk_idx][b] - i_feat_value;
                        end
                    end
                end

                S_CLEARING: begin
                    // Clear one chunk per cycle
                    for (int b = 0; b < CHUNK_WIDTH; b++)
                        acc[op_chunk_cnt][b] <= '0;

                    if (op_chunk_cnt == CHUNK_IDX_W'(NUM_CHUNKS - 1)) begin
                        state_q <= S_IDLE;
                    end else begin
                        op_chunk_cnt <= op_chunk_cnt + 1'b1;
                    end
                end

                S_BINARIZE: begin
                    // Binarize one chunk per cycle
                    o_qhv_valid     <= 1'b1;
                    o_qhv_chunk_idx <= op_chunk_cnt;
                    for (int b = 0; b < CHUNK_WIDTH; b++)
                        o_qhv_chunk_data[b] <= (acc[op_chunk_cnt][b] >= 0) ? 1'b1 : 1'b0;

                    if (op_chunk_cnt == CHUNK_IDX_W'(NUM_CHUNKS - 1)) begin
                        state_q         <= S_IDLE;
                        o_binarize_done <= 1'b1;
                    end else begin
                        op_chunk_cnt <= op_chunk_cnt + 1'b1;
                    end
                end

                default: state_q <= S_IDLE;
            endcase
        end
    end

endmodule
