TOPLEVEL_LANG ?= verilog
SIM ?= verilator

export VERILATOR_NO_CCACHE ?= 1
export CCACHE :=
export VERILATOR_CXX ?= g++
export CCACHE_DIR ?= $(PWD)/.ccache
export TRACE ?= 1
export WAVEFORM_FILE ?= dump.vcd
export EXTRA_ARGS += --trace --trace-structs
export SIM_ARGS += --trace --trace-file $(WAVEFORM_FILE)

TOPLEVEL := tiny_bnn_accel
MODULE   := testbench
VERILOG_SOURCES := $(PWD)/model.sv

COCOTB_CONFIG ?= $(shell if command -v cocotb-config >/dev/null 2>&1; then command -v cocotb-config; elif [ -x "$(PWD)/../.venv/bin/cocotb-config" ]; then echo $(PWD)/../.venv/bin/cocotb-config; else echo cocotb-config; fi)

include $(shell $(COCOTB_CONFIG) --makefiles)/Makefile.sim
