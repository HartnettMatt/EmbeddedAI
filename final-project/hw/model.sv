// Tiny binarized NN accelerator with XNOR-popcount + threshold + argmax.

// Synthesis-friendly ROM content provided via generated include.
`include "model.svh"

module tiny_bnn_accel #(
    parameter int unsigned DATA_WIDTH = 64,  // bits per incoming image payload
    parameter int unsigned TID_WIDTH  = 4
) (
    input logic clk,
    input logic rst_n,

    // Slave AXI-Stream input (image payload)
    input  logic [DATA_WIDTH-1:0] s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic [ TID_WIDTH-1:0] s_axis_tid,

    // Master AXI-Stream output (classification result)
    output logic [          7:0] m_axis_tdata,
    output logic                 m_axis_tvalid,
    input  logic                 m_axis_tready,
    output logic [TID_WIDTH-1:0] m_axis_tid
);

  localparam int unsigned L1_IN = DATA_WIDTH;
  localparam int unsigned L1_OUT = 128;
  localparam int unsigned L2_IN = L1_OUT;
  localparam int unsigned L2_OUT = 10;

  localparam int unsigned L1_W_BITS = L1_IN * L1_OUT;
  localparam int unsigned L1_W_WORDS = (L1_W_BITS + 31) / 32;
  localparam int unsigned L1_B_WORDS = L1_OUT;
  localparam int unsigned L1_T_WORDS = L1_OUT;
  localparam int unsigned L1_S_WORDS = (L1_OUT + 31) / 32;
  localparam int unsigned L2_W_BITS = L2_IN * L2_OUT;
  localparam int unsigned L2_W_WORDS = (L2_W_BITS + 31) / 32;
  localparam int unsigned L2_B_WORDS = L2_OUT;

  localparam int unsigned MEM_DEPTH = L1_W_WORDS + L1_B_WORDS + L1_T_WORDS + L1_S_WORDS
                                      + L2_W_WORDS + L2_B_WORDS;

  localparam int unsigned L1_W_BASE = 0;
  localparam int unsigned L1_B_BASE = L1_W_BASE + L1_W_WORDS;
  localparam int unsigned L1_T_BASE = L1_B_BASE + L1_B_WORDS;
  localparam int unsigned L1_S_BASE = L1_T_BASE + L1_T_WORDS;
  localparam int unsigned L2_W_BASE = L1_S_BASE + L1_S_WORDS;
  localparam int unsigned L2_B_BASE = L2_W_BASE + L2_W_WORDS;

  logic [L1_OUT-1:0] hidden_bits_comb;
  logic [       7:0] pred_class_comb;

  logic [L1_OUT-1:0] hidden_bits_q;
  logic [       7:0] pred_class_q;

    assign s_axis_tready = ~m_axis_tvalid;

  // Compute hidden layer (XNOR-popcount >= threshold).
  always_comb begin
    hidden_bits_comb = '0;
    for (int o = 0; o < L1_OUT; o++) begin
      int match_count;
      int signed thr_val;
      match_count = 0;
      for (int i = 0; i < L1_IN; i++) begin
        int   idx;
        int   word_idx;
        int   bit_idx;
        logic weight_bit;
        idx        = i * L1_OUT + o;
        word_idx   = idx / 32;
        bit_idx    = idx % 32;
        weight_bit = MODEL_MEM[L1_W_BASE+word_idx][bit_idx];
        if (s_axis_tdata[i] == weight_bit) begin
          match_count++;
        end
      end
      thr_val = int'($signed(MODEL_MEM[L1_T_BASE+o][15:0]));
      begin
        int   sense_idx;
        int   sense_word;
        int   sense_bit;
        logic sense_pos;
        sense_idx  = o;
        sense_word = sense_idx / 32;
        sense_bit  = sense_idx % 32;
        sense_pos  = MODEL_MEM[L1_S_BASE+sense_word][sense_bit];
        if (sense_pos) begin
          hidden_bits_comb[o] = ($signed(match_count) >= thr_val);
        end else begin
          hidden_bits_comb[o] = ($signed(match_count) <= thr_val);
        end
      end
    end
  end

  // Compute output layer and argmax.
  always_comb begin
    int signed best_score;
    int best_class;
    best_score = -32'sd2000000000;
    best_class = 0;
    for (int c = 0; c < L2_OUT; c++) begin
      int match_count;
      int signed bias_q;
      int signed score;
      match_count = 0;
      for (int i = 0; i < L2_IN; i++) begin
        int   idx;
        int   word_idx;
        int   bit_idx;
        logic weight_bit;
        idx        = i * L2_OUT + c;
        word_idx   = idx / 32;
        bit_idx    = idx % 32;
        weight_bit = MODEL_MEM[L2_W_BASE+word_idx][bit_idx];
        if (hidden_bits_comb[i] == weight_bit) begin
          match_count++;
        end
      end
      bias_q = int'($signed(MODEL_MEM[L2_B_BASE+c][15:0]));
      score  = (match_count * 2 - L2_IN) + bias_q;
      if (score > best_score) begin
        best_score = score;
        best_class = c;
      end
    end
    pred_class_comb = best_class[7:0];
  end

  // Registers and handshake.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      hidden_bits_q <= '0;
      pred_class_q  <= '0;
      m_axis_tid    <= '0;
      m_axis_tdata  <= '0;
      m_axis_tvalid <= 1'b0;
    end else begin
      if (m_axis_tvalid && m_axis_tready) begin
        m_axis_tvalid <= 1'b0;
      end

      if (s_axis_tvalid && s_axis_tready) begin
        hidden_bits_q <= hidden_bits_comb;
        pred_class_q  <= pred_class_comb;
        m_axis_tid    <= s_axis_tid;
        m_axis_tdata  <= pred_class_comb;
        m_axis_tvalid <= 1'b1;
      end
    end
  end

endmodule
