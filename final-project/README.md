# Final Project: FPGA TinyNN in SystemVerilog

Matt Hartnett  
ECEN 5003: Embedded AI 

## Guide

Required Tools:
- SystemVerilog
- Verilator
- UV
- Python
- cocotb

Environment setup:
1. Run `uv sync`
2. Run `source .venv/bin/activate`

Running model:
**TBD**

Running testbench:
**TBD**

## Goal & Scope

Primary goal: Build and verify a tiny, binarized neural network (BNN) accelerator in SystemVerilog that performs image classification using XNOR–popcount compute. 

Stretch goal: Add convolutional layers to the neural network to improve accuracy.

Deliverable: 
- `rtl/` SystemVerilog accelerator (perceptron + optional CNV-1)
- `sim/` cocotb tests
- `model/` Python training + export
- `docs/` final report
- Evidence: waveforms, latency/throughput numbers, accuracy vs. Python golden, and a short demo clip.

---

## Data, Ingestion, and Processing

The dataset will be based on `scikit-learn digits` (8×8 grayscale images).

Ingestion:
**TBD**

Processing:
**TBD**

---

## Hardware Architecture

Top-level:
- Interface: stream in/stream out port for images
- On-chip memories: parameterizable BRAM for weights/thresholds.  
- Timing: single-image, deterministic latency

Datapath:
**TBD**

---

## Verification Strategy (Verilator + cocotb)

- Vector tests: feed many pre-exported images; compare class to Python golden  
- Corner cases: all-zeros, all-ones, single-pixel set, random noise 
- Latency/throughput: measure cycles per image  
- Back-to-back images: ensure handshake correctness and no state leakage

---

## Plan for Completion

Week 1
1. Data & training (Python): train minimal model; export weights/thresholds + 200 test images + goldens.
2. RTL skeleton: top-level streams, BRAM init from `.memh`, XNOR–popcount slice, threshold compare, argmax.
3. Basic test (cocotb): single image w/out class checking to verify IO

---

Week 2

4. Robust testbench: randomized batches, corner cases, coverage of all classes.  
5. Performance tuning: widen slice width; pipeline popcount; measure cycles/image.  

---

Week 3

6. Stretch Goal: add convolution, re-run tests
8. Polish & deliver: finalize docs, figures, and a short demo clip of sim run.

---