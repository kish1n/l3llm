#pragma once

// The whole forward pass:
//
//   embed -> 16 x decoder_layer -> model.norm -> lm_head -> logits
//
// Everything position-dependent lives inside the layers (RoPE on Q and K),
// so this level is just a loop over a residual stream.

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "config.h"
#include "safetensors.h"

namespace qllm {

struct ForwardResult {
    std::vector<float> hidden; // after model.norm, [n_tokens, hidden_size]
    std::vector<float> logits; // [n_tokens, vocab_size]
};

// Called after each decoder layer with the current residual stream,
// mirroring the per-layer hooks dump_golden.py registers. Optional.
using LayerHook = std::function<void(std::int64_t index, std::span<const float> x)>;

// With tie_word_embeddings the output projection reuses
// model.embed_tokens.weight, so row i of that matrix is both token i's
// input embedding and the direction giving token i's logit.
ForwardResult forward(const SafeTensors &weights, const ModelConfig &cfg,
                      std::span<const std::int32_t> token_ids, const LayerHook &on_layer = {});

// Greedy next token: argmax over the last row of logits. Ties go to the
// lower id, matching torch.argmax.
std::int32_t argmax_last(const ForwardResult &out, const ModelConfig &cfg);

} // namespace qllm
