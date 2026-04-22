// =============================================================================
// Module : hamming_sim
// Project: HDC-FPGA Inference Accelerator
// Lang   : SystemVerilog (IEEE 1800-2012)
// Brief  : sim[c] = popcount(hv_query XNOR centroid[c])  for c = 0..C-1.
//          Sequential: issues one BRAM read per cycle, pipelines read→compute.
//          Total latency after hv_valid = C + 1 cycles (1 for BRAM latency,
//          C for C centroid computations), then scores_valid fires.
// =============================================================================
`timescale 1ns/1ps

module hamming_sim #(
    parameter int D       = 4096,
    parameter int C       = 6,
    parameter int ADDR_W  = 3,
    parameter int SCORE_W = 13
)(
    input  logic               clk,
    input  logic               rst_n,
    // from binarizer
    input  logic [D-1:0]       hv_query,
    input  logic               hv_valid,
    // centroid BRAM interface
    output logic [ADDR_W-1:0]  rd_addr,
    output logic               rd_en,
    input  logic [D-1:0]       centroid,   // registered BRAM output, 1-cycle latency
    // result
    output logic [C*SCORE_W-1:0] score_out,
    output logic                 scores_valid,
    output logic                 busy
);

    // ── FSM ───────────────────────────────────────────────────────────────────
    // Pipeline: IDLE → READ (issue read) →
    //           WAIT  (BRAM latency + latch centroid) →
    //           POP   (iterative popcount across CHUNK bits per cycle) →
    //           DONE  (assert scores_valid for one cycle) → IDLE
    typedef enum logic [2:0] {
        S_IDLE  = 3'd0,
        S_READ  = 3'd1,
        S_WAIT  = 3'd2,
        S_LATCH = 3'd3,
        S_POP   = 3'd4,
        S_DONE  = 3'd5
    } state_t;

    state_t               state;
    logic [ADDR_W-1:0]    cls_cnt;     // next class to READ
    logic [ADDR_W-1:0]    pend_cls;    // class whose BRAM read is in-flight
    logic [D-1:0]         query_reg;   // latched query HV
    logic [D-1:0]         centroid_reg;

    localparam int CHUNK     = 64;
    localparam int CHUNK_CNT = (D + CHUNK - 1) / CHUNK;
    localparam int CHUNK_W   = (CHUNK_CNT <= 1) ? 1 : $clog2(CHUNK_CNT);

    logic [CHUNK_W-1:0]    chunk_idx;
    logic [SCORE_W-1:0]    pop_accum;
    logic [SCORE_W-1:0]    chunk_pop;
    logic [SCORE_W-1:0]    pop_next;

    always_comb begin
        chunk_pop = '0;
        for (int b = 0; b < CHUNK; b++) begin
            int gi;
            logic xnor_bit;
            gi = (chunk_idx * CHUNK) + b;
            xnor_bit = 1'b0;
            if (gi < D)
                xnor_bit = ~(query_reg[gi] ^ centroid_reg[gi]);
            chunk_pop = chunk_pop + {{(SCORE_W-1){1'b0}}, xnor_bit};
        end
    end

    assign pop_next = pop_accum + chunk_pop;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            cls_cnt      <= '0;
            pend_cls     <= '0;
            rd_addr      <= '0;
            rd_en        <= 1'b0;
            score_out    <= '0;
            scores_valid <= 1'b0;
            query_reg    <= '0;
            centroid_reg <= '0;
            chunk_idx    <= '0;
            pop_accum    <= '0;
        end else begin
            rd_en        <= 1'b0;    // default de-assert
            scores_valid <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    if (hv_valid) begin
                        query_reg <= hv_query;
                        cls_cnt   <= '0;
                        score_out <= '0;
                        state     <= S_READ;
                    end
                end

                // issue BRAM read for class cls_cnt
                S_READ: begin
                    rd_addr  <= cls_cnt;
                    rd_en    <= 1'b1;
                    pend_cls <= cls_cnt;
                    cls_cnt  <= cls_cnt + 1'b1;
                    state    <= S_WAIT;
                end

                // wait one cycle for BRAM registered output to settle
                S_WAIT: begin
                    state <= S_LATCH;
                end

                S_LATCH: begin
                    centroid_reg <= centroid;
                    chunk_idx    <= '0;
                    pop_accum    <= '0;
                    state        <= S_POP;
                end

                S_POP: begin
                    pop_accum <= pop_next;
                    if (chunk_idx == CHUNK_W'(CHUNK_CNT - 1)) begin
                        score_out[pend_cls * SCORE_W +: SCORE_W] <= pop_next;
                        if (pend_cls == ADDR_W'(C - 1)) begin
                            state <= S_DONE;
                        end else begin
                            state <= S_READ;
                        end
                    end else begin
                        chunk_idx <= chunk_idx + 1'b1;
                    end
                end

                S_DONE: begin
                    scores_valid <= 1'b1;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

endmodule
