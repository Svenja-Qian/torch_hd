// =============================================================================
// Module : projection_engine
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : y[d] = SUM_k (W[d][k] * x[k]),  W[d][k] ∈ {+1, −1}.
//          Processes one (dim, feature) pair per clock.
//          Latency = D × FEAT_NUM clocks after frame_valid.
// =============================================================================
`timescale 1ns/1ps

module projection_engine #(
    parameter int D        = 4096,
    parameter int FEAT_NUM = 561,
    parameter int DATA_W   = 16,
    parameter int ACC_W    = 26,
    parameter      WEIGHT_INIT_FILE = "",
    parameter bit  ALLOW_DUMMY_WEIGHTS = 1'b1
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic [FEAT_NUM*DATA_W-1:0]   feature_vec,
    input  logic                         frame_valid,
    output logic [D*ACC_W-1:0]           acc_out,
    output logic                         proj_valid,
    output logic                         busy
);

    // ── boundary constants (Fix-2) ────────────────────────────────────────────
    localparam int FEAT_LAST = FEAT_NUM - 1;
    localparam int DIM_LAST  = D - 1;

    // ── unpack feature_vec ────────────────────────────────────────────────────
    // frame_ctrl places feature[0] at the MSB end of feature_vec:
    //   feature[gi] → feature_vec[(FEAT_NUM-1-gi)*DATA_W +: DATA_W]
    logic signed [DATA_W-1:0] feat [0:FEAT_NUM-1];
    for (genvar gi = 0; gi < FEAT_NUM; gi++) begin : g_unpack
        assign feat[gi] = feature_vec[(FEAT_NUM-1-gi)*DATA_W +: DATA_W];
    end

    // ── weight ROM ────────────────────────────────────────────────────────────
    // w_rom[d][k]: 1 → +1,  0 → −1
    logic [FEAT_NUM-1:0] w_rom [0:D-1];
    initial begin
        if (WEIGHT_INIT_FILE != "") begin
`ifndef SYNTHESIS
            int init_fd;
            init_fd = $fopen(WEIGHT_INIT_FILE, "r");
            if (init_fd == 0) begin
                $fatal(1, "projection_engine could not open WEIGHT_INIT_FILE='%s'", WEIGHT_INIT_FILE);
            end
            $fclose(init_fd);
`endif
            $readmemb(WEIGHT_INIT_FILE, w_rom);
        end else if (ALLOW_DUMMY_WEIGHTS) begin
            for (int d_i = 0; d_i < D; d_i++) begin
                for (int k_i = 0; k_i < FEAT_NUM; k_i++) begin
                    int idx;
                    idx = (d_i * FEAT_NUM) + k_i;
                    w_rom[d_i][k_i] = ^idx[9:0];
                end
            end
        end else begin
            for (int d_i = 0; d_i < D; d_i++)
                w_rom[d_i] = '0;
`ifndef SYNTHESIS
            $fatal(1, "projection_engine WEIGHT_INIT_FILE is empty but ALLOW_DUMMY_WEIGHTS=0");
`endif
        end
    end

    // ── FSM ───────────────────────────────────────────────────────────────────
    typedef enum logic { S_IDLE = 1'b0, S_CALC = 1'b1 } state_t;

    state_t                      state;
    logic [$clog2(FEAT_NUM)-1:0] feat_cnt;
    logic [$clog2(D)-1:0]        dim_cnt;
    logic signed [ACC_W-1:0]     acc_cur;

    // ── combinational signals ─────────────────────────────────────────────────
    // Explicit sign-extend to ACC_W bits (Fix-3)
    logic signed [ACC_W-1:0] addend;
    assign addend = {{(ACC_W-DATA_W){feat[feat_cnt][DATA_W-1]}}, feat[feat_cnt]};

    logic                    dim_done;
    logic                    last_dim;
    logic signed [ACC_W-1:0] dim_result;
    logic [$clog2(FEAT_NUM)-1:0] feat_last_w;
    logic [$clog2(D)-1:0]        dim_last_w;

    assign feat_last_w = FEAT_LAST[$clog2(FEAT_NUM)-1:0];
    assign dim_last_w  = DIM_LAST[$clog2(D)-1:0];

    assign dim_done   = (state == S_CALC) && (feat_cnt == feat_last_w);
    assign last_dim   = dim_done && (dim_cnt == dim_last_w);
    assign dim_result = w_rom[dim_cnt][feat_cnt]
                        ? acc_cur + addend
                        : acc_cur - addend;

    // ── FSM sequential ────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            feat_cnt <= '0;
            dim_cnt  <= '0;
            acc_cur  <= '0;
        end else begin
            unique case (state)
                S_IDLE: begin
                    if (frame_valid) begin
                        dim_cnt  <= '0;
                        feat_cnt <= '0;
                        acc_cur  <= '0;
                        state    <= S_CALC;
                    end
                end

                S_CALC: begin
                    if (feat_cnt == feat_last_w) begin
                        feat_cnt <= '0;
                        acc_cur  <= '0;
                        if (dim_cnt == dim_last_w) begin
                            dim_cnt <= '0;
                            state   <= S_IDLE;
                        end else begin
                            dim_cnt <= dim_cnt + 1'b1;
                        end
                    end else begin
                        acc_cur  <= w_rom[dim_cnt][feat_cnt]
                                    ? acc_cur + addend
                                    : acc_cur - addend;
                        feat_cnt <= feat_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

    // ── output: direct indexed write (Fix-1) ─────────────────────────────────
    // acc_out[d*ACC_W +: ACC_W] holds the accumulator for dim d.
    // proj_valid fires for exactly one cycle when the last dim is committed.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out    <= '0;
            proj_valid <= 1'b0;
        end else begin
            proj_valid <= 1'b0;
            if (dim_done) begin
                acc_out[dim_cnt * ACC_W +: ACC_W] <= dim_result;
                proj_valid <= last_dim;
            end
        end
    end

endmodule
