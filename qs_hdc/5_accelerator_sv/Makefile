# =============================================================================
# Makefile - HDC FPGA Accelerator (SystemVerilog)
# Default simulator: Icarus Verilog (iverilog -g2012)
#
# Supports both layouts:
# 1) flat files in project root (default)
# 2) separated folders via: make RTL_DIR=rtl TB_DIR=tb <target>
# =============================================================================

IVERILOG ?= iverilog
VVP      ?= vvp
VFLAGS   ?= -g2012 -Wall
RM       ?= rm -f

RTL_DIR  ?= .
TB_DIR   ?= .

RTL_ALL  := $(RTL_DIR)/uart_rx.sv            \
            $(RTL_DIR)/uart_tx.sv            \
            $(RTL_DIR)/frame_ctrl.sv         \
            $(RTL_DIR)/projection_engine.sv  \
            $(RTL_DIR)/binarizer.sv          \
            $(RTL_DIR)/centroid_bram.sv      \
            $(RTL_DIR)/hamming_sim.sv        \
            $(RTL_DIR)/argmax.sv             \
            $(RTL_DIR)/hdc_top.sv

.PHONY: all clean \
        sim_uart_rx sim_uart_tx sim_frame_ctrl \
        sim_binarizer sim_projection_engine sim_hamming_argmax sim_top sim_ucihar_core

all: sim_uart_rx sim_uart_tx sim_frame_ctrl sim_binarizer sim_projection_engine sim_hamming_argmax sim_top

sim_uart_rx: $(TB_DIR)/tb_uart_rx.sv $(RTL_DIR)/uart_rx.sv
	@echo ">>> [1/6] uart_rx"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_uart_tx: $(TB_DIR)/tb_uart_tx.sv $(RTL_DIR)/uart_tx.sv $(RTL_DIR)/uart_rx.sv
	@echo ">>> [2/6] uart_tx"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_frame_ctrl: $(TB_DIR)/tb_frame_ctrl.sv $(RTL_DIR)/frame_ctrl.sv
	@echo ">>> [3/6] frame_ctrl"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_binarizer: $(TB_DIR)/tb_binarizer.sv $(RTL_DIR)/binarizer.sv
	@echo ">>> [4/6] binarizer"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_projection_engine: $(TB_DIR)/tb_projection_engine.sv $(RTL_DIR)/projection_engine.sv
	@echo ">>> [5/7] projection_engine"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_hamming_argmax: $(TB_DIR)/tb_hamming_argmax.sv \
                    $(RTL_DIR)/hamming_sim.sv \
                    $(RTL_DIR)/argmax.sv \
                    $(RTL_DIR)/centroid_bram.sv
	@echo ">>> [6/7] hamming_sim + argmax + centroid_bram"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_top: $(TB_DIR)/tb_hdc_top.sv $(RTL_ALL)
	@echo ">>> [7/7] hdc_top E2E"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

sim_ucihar_core: $(TB_DIR)/tb_ucihar_core.sv $(RTL_ALL)
	@echo ">>> [EXTRA] ucihar core (no UART)"
	$(IVERILOG) $(VFLAGS) -o $@.vvp $^
	$(VVP) $@.vvp
	@echo

clean:
	$(RM) *.vvp *.vcd
