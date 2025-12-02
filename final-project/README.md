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
1. `python ./sw/model.py`
Alternatively, run interactively as an ipynb in VSCode or online (section delimiters are `# %%`)

Running testbench:
1. `cd hw/`
2. `make`
3. `gtkwave ./dump.vcd -a ./tb.gtkw`


## Goal & Scope

Primary goal: Build and verify a tiny, binarized neural network (BNN) accelerator in SystemVerilog that performs image classification using XNOR–popcount compute. 

Stretch goal: Add convolutional layers to the neural network to improve accuracy.

Deliverable: 
- `hw/` SystemVerilog model + cocotb simulation
- `sw/` Python training + export
- `docs/` final report
- Evidence: waveforms, latency/throughput numbers, accuracy vs. Python golden, and a short demo clip.

---

## Data, Ingestion, and Processing

The dataset will be based on `scikit-learn digits` (8×8 grayscale images).

Ingestion:
- scikit-learn digits dataset is cached to `sw/dataset/digits.npz` on first run of `sw/model.py`.

Processing:
- Normalize images to [-1,1], binarize to {0,1}, train/export a two-layer BinaryDense network; artifacts include packed weights, thresholds, sense bits, and golden vectors.

---

## Hardware Architecture

Top-level:
- Interface: stream in/stream out port for images
- On-chip memories: parameterizable BRAM for weights/thresholds.  
- Timing: single-image, deterministic latency

Datapath:
- AXI-Stream in (64-bit image) 
- XNOR-popcount hidden layer (128 neurons, sense-aware threshold) 
- binary output layer (10 classes) with match-count + bias 
- argmax 
- AXI-Stream out (class byte + TID). ROM weights/thresholds come from `model.svh`.

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
2. RTL skeleton: top-level streams, BRAM init from `.svh`, XNOR–popcount slice, threshold compare, argmax.
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

## Use of Generative AI

Generative AI was utilized in this project, primarily for code cleanup and documentation.

Model: OpenAI ChatGPT 5.1 Thinking
Prompt: For this python/systemverilog file, provide a concise, precise list of potential cleanups, documentation/comments, and optimizations that can be made. For each improvement, provide a drop in code snippet with minimal edits as well as the exact location of the code where the snippet can be placed.
