//==============================================================================
// feature_scheduler.sv
// Sequentially reads features from input_buffer and dispatches them to the
// 3 experts according to the current top-level FSM state and phase counts.
//
// Phases:
//   S_SHARED_ENC     : expert_en = 3'b111, broadcast to all experts.
//                      Wait for ALL 3 chunks_done before advancing.
//   S_PRIVATE_ENC_e  : expert_en = one-hot for expert e.
//                      Wait for expert e's chunks_done before advancing.
//
// Phase end: when phase_feat_cnt reaches the corresponding i_num_*_feat value.
// Phase count == 0: immediately raise phase_done (skip).
//
// Sequential read only — never reorders or skips FIFO entries.
//
// Handshake to experts:
//   o_feat_valid pulses high for 1 cycle per dispatched feature.
//   Scheduler waits for i_chunks_done[e] (for enabled experts) before
//   advancing to the next feature. i_acc_ready gates dispatch (assertion).
//==============================================================================

module feature_scheduler #(
    parameter int unsigned NUM_EXPERTS  = 3,
    parameter int unsigned FEAT_WIDTH   = 16,
    parameter int unsigned FEAT_IDX_W   = 8,
    parameter int unsigned FEAT_CNT_W   = 9,
    parameter int unsigned FSM_STATE_W  = 4,

    // FSM state encodings (must match top_controller_fsm)
    parameter logic [FSM_STATE_W-1:0] S_SHARED_ENC    = 4'd2,
    parameter logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_0 = 4'd3,
    parameter logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_1 = 4'd4,
    parameter logic [FSM_STATE_W-1:0] S_PRIVATE_ENC_2 = 4'd5
) (
    input  logic                                    clk,
    input  logic                                    rst_n,

    // FSM interface
    input  logic [FSM_STATE_W-1:0]                  i_state,

    // Phase count configuration (valid before i_start at top level)
    input  logic [FEAT_CNT_W-1:0]                   i_num_shared_feat,
    input  logic [FEAT_CNT_W-1:0]                   i_num_priv_feat_0,
    input  logic [FEAT_CNT_W-1:0]                   i_num_priv_feat_1,
    input  logic [FEAT_CNT_W-1:0]                   i_num_priv_feat_2,

    // Read side of input_buffer
    input  logic                                    i_ibuf_empty,
    input  logic signed [FEAT_WIDTH-1:0]            i_ibuf_feat_value,
    input  logic [FEAT_IDX_W-1:0]                   i_ibuf_feat_index,
    output logic                                    o_ibuf_rd_en,

    // Dispatch to projection_bram_interface / accumulator_bank
    output logic [NUM_EXPERTS-1:0]                  o_expert_en,
    output logic                                    o_feat_valid,
    output logic signed [FEAT_WIDTH-1:0]            o_feat_value,
    output logic [FEAT_IDX_W-1:0]                   o_feat_index,

    // Synchronization feedback from experts
    input  logic [NUM_EXPERTS-1:0]                  i_chunks_done,
    input  logic [NUM_EXPERTS-1:0]                  i_acc_ready,

    // Phase completion to top FSM
    output logic                                    o_shared_phase_done,
    output logic [NUM_EXPERTS-1:0]                  o_private_phase_done
);

    // -------------------------------------------------------------------------
    // Internal substate: 
    //   WAIT_FEAT   -> feature pending dispatch
    //   DISPATCH    -> one-cycle feat_valid pulse
    //   WAIT_CHUNKS -> wait for required chunks_done
    //   DONE_PHASE  -> phase complete (until FSM moves on)
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        SUB_WAIT_FEAT   = 2'd0,
        SUB_DISPATCH    = 2'd1,
        SUB_WAIT_CHUNKS = 2'd2,
        SUB_DONE_PHASE  = 2'd3
    } sub_state_t;

    sub_state_t              sub_q;
    logic [FEAT_CNT_W-1:0]   phase_feat_cnt_q;
    logic [FSM_STATE_W-1:0]  prev_state_q;

    // -------------------------------------------------------------------------
    // Decode the current phase from FSM state
    // -------------------------------------------------------------------------
    wire is_shared   = (i_state == S_SHARED_ENC);
    wire is_priv_0   = (i_state == S_PRIVATE_ENC_0);
    wire is_priv_1   = (i_state == S_PRIVATE_ENC_1);
    wire is_priv_2   = (i_state == S_PRIVATE_ENC_2);
    wire is_any_enc  = is_shared | is_priv_0 | is_priv_1 | is_priv_2;
    wire is_private  = is_priv_0 | is_priv_1 | is_priv_2;

    // Current phase's target feature count
    logic [FEAT_CNT_W-1:0] phase_target;
    always_comb begin
        unique case (1'b1)
            is_shared:  phase_target = i_num_shared_feat;
            is_priv_0:  phase_target = i_num_priv_feat_0;
            is_priv_1:  phase_target = i_num_priv_feat_1;
            is_priv_2:  phase_target = i_num_priv_feat_2;
            default:    phase_target = '0;
        endcase
    end

    // Expert enable mask for the current phase
    logic [NUM_EXPERTS-1:0] expert_en_phase;
    always_comb begin
        unique case (1'b1)
            is_shared:  expert_en_phase = {NUM_EXPERTS{1'b1}};      // 3'b111
            is_priv_0:  expert_en_phase = 3'b001;
            is_priv_1:  expert_en_phase = 3'b010;
            is_priv_2:  expert_en_phase = 3'b100;
            default:    expert_en_phase = '0;
        endcase
    end

    // -------------------------------------------------------------------------
    // Synchronization predicate — all required experts' chunks_done observed
    // -------------------------------------------------------------------------
    // We latch chunks_done pulses on a per-expert basis during WAIT_CHUNKS,
    // in case experts finish in different cycles.
    logic [NUM_EXPERTS-1:0] chunks_done_seen_q;
    wire  [NUM_EXPERTS-1:0] chunks_done_seen_next = chunks_done_seen_q | i_chunks_done;

    wire all_required_done =
        ((chunks_done_seen_next & expert_en_phase) == expert_en_phase);

    // -------------------------------------------------------------------------
    // acc_ready check for enabled experts — must be all-ready before dispatch
    // -------------------------------------------------------------------------
    wire acc_ready_for_phase =
        ((i_acc_ready & expert_en_phase) == expert_en_phase);

    // -------------------------------------------------------------------------
    // Phase-enter detection (state change)
    // -------------------------------------------------------------------------
    wire state_changed   = (i_state != prev_state_q);
    wire phase_count_zero = (phase_target == '0);

    // -------------------------------------------------------------------------
    // Output defaults
    // -------------------------------------------------------------------------
    always_comb begin
        o_expert_en  = '0;
        o_feat_valid = 1'b0;
        o_feat_value = '0;
        o_feat_index = '0;
        o_ibuf_rd_en = 1'b0;

        o_shared_phase_done  = 1'b0;
        o_private_phase_done = '0;

        if (is_any_enc) begin
            // Expert enable mask stays asserted through the whole phase so that
            // downstream projection_bram_interface stays enabled. It's OK
            // because feat_valid is the only pulse that matters for dispatch.
            o_expert_en = expert_en_phase;

            if (sub_q == SUB_DISPATCH) begin
                o_feat_valid = 1'b1;
                o_feat_value = i_ibuf_feat_value;
                o_feat_index = i_ibuf_feat_index;
                o_ibuf_rd_en = 1'b1;        // consume this FIFO entry now
            end

            // Emit phase_done ONLY if sub_q has stably reached SUB_DONE_PHASE
            // for the CURRENT phase. In the first cycle after a state change,
            // sub_q may still be SUB_DONE_PHASE from the previous phase — gate
            // with !state_changed to prevent a false done pulse.
            if (sub_q == SUB_DONE_PHASE && !state_changed) begin
                if (is_shared) o_shared_phase_done = 1'b1;
                if (is_priv_0) o_private_phase_done[0] = 1'b1;
                if (is_priv_1) o_private_phase_done[1] = 1'b1;
                if (is_priv_2) o_private_phase_done[2] = 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sub_q               <= SUB_WAIT_FEAT;
            phase_feat_cnt_q    <= '0;
            chunks_done_seen_q  <= '0;
            prev_state_q        <= '0;
        end else begin
            prev_state_q <= i_state;

            // ----- Phase entry: reset counters -----
            if (state_changed && is_any_enc) begin
                phase_feat_cnt_q   <= '0;
                chunks_done_seen_q <= '0;
                if (phase_count_zero) begin
                    // Immediately skip this phase
                    sub_q <= SUB_DONE_PHASE;
                end else begin
                    sub_q <= SUB_WAIT_FEAT;
                end
            end
            // ----- Leaving any encode phase: return to idle sub_state -----
            else if (state_changed && !is_any_enc) begin
                sub_q               <= SUB_WAIT_FEAT;
                phase_feat_cnt_q    <= '0;
                chunks_done_seen_q  <= '0;
            end
            // ----- Normal substate evolution -----
            else if (is_any_enc) begin
                unique case (sub_q)
                    SUB_WAIT_FEAT: begin
                        // Wait for FIFO non-empty AND all required experts ready
                        if (!i_ibuf_empty && acc_ready_for_phase) begin
                            sub_q <= SUB_DISPATCH;
                        end
                    end

                    SUB_DISPATCH: begin
                        // One-cycle pulse: feat_valid high, rd_en high this cycle
                        // Advance to wait for chunks_done next cycle
                        chunks_done_seen_q <= '0; // clear for this feature
                        sub_q              <= SUB_WAIT_CHUNKS;
                    end

                    SUB_WAIT_CHUNKS: begin
                        // Latch chunks_done pulses from each expert
                        chunks_done_seen_q <= chunks_done_seen_next;

                        if (all_required_done) begin
                            chunks_done_seen_q <= '0;
                            phase_feat_cnt_q   <= phase_feat_cnt_q + 1'b1;

                            if ((phase_feat_cnt_q + 1'b1) == phase_target) begin
                                sub_q <= SUB_DONE_PHASE;
                            end else begin
                                sub_q <= SUB_WAIT_FEAT;
                            end
                        end
                    end

                    SUB_DONE_PHASE: begin
                        // Hold here; top FSM will transition state and we'll
                        // re-init via state_changed path.
                    end

                    default: sub_q <= SUB_WAIT_FEAT;
                endcase
            end
        end
    end

endmodule
