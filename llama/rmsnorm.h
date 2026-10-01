#pragma once

// Root-mean-square layer norm, applied per token over the hidden dim:
//
//   y = gamma * x / sqrt(mean(x^2) + eps)
//
// Not LayerNorm: there is no mean subtraction and no bias, so the result
// has RMS 1 before the gain rather than mean 0 / variance 1. Hugging Face
// still names the weights "input_layernorm" and "post_attention_layernorm";
// both are this.

#include <cstddef>
#include <span>
#include <vector>

#include "safetensors.h"

namespace l3llm {

// x is [n_tokens, hidden] row-major, gamma the [hidden] BF16 gain from the
// weights file, eps the config's rms_norm_eps. Returns a fresh buffer: the
// caller still needs x intact for the residual add.
std::vector<float> rmsnorm(std::span<const float> x, const TensorView &gamma, double eps);

} // namespace l3llm
