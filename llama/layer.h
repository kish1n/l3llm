#pragma once

// One transformer decoder layer, applied in place to the residual stream.
//
//   x = x + attn(rmsnorm(x, input_layernorm))
//   x = x + mlp (rmsnorm(x, post_attention_layernorm))
//
// Pre-norm: each norm sits inside a branch, never on the trunk, so the
// residual path stays an unobstructed identity. The second norm reads x
// after the first add, not the original input. Despite the name,
// post_attention_layernorm normalizes the MLP's input.

#include <cstdint>
#include <span>

#include "config.h"
#include "rope.h"
#include "safetensors.h"

namespace l3llm {

// x is [n_tokens, hidden] and is updated in place. `table` is shared by
// every layer -- RoPE angles depend only on the config.
void decoder_layer(std::span<float> x, const SafeTensors &weights, const ModelConfig &cfg,
                   std::int64_t index, const RopeTable &table, std::int64_t n_tokens);

} // namespace l3llm
