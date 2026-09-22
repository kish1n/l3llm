#pragma once

// Bias-free linear projection -- the one matmul every weight matrix in the
// model goes through: q/k/v/o in attention, gate/up/down in the MLP.
//
// Hugging Face stores nn.Linear weights as [out_features, in_features], so
// the op is y = x @ W.T:
//
//   y[t][o] = sum_i x[t][i] * W[o][i]
//
// Both operands of that dot product are contiguous rows, which is why no
// transpose is needed anywhere in the forward pass.

#include <span>
#include <vector>

#include "safetensors.h"

namespace qllm {

// x is [n_tokens, in] float32 row-major, w the [out, in] BF16 weight.
// Returns [n_tokens, out] float32.
//
// Accumulates in float, matching the fp32 BLAS the reference dispatches to.
// Widening happens in registers: the weight matrix is never materialized.
std::vector<float> linear(std::span<const float> x, const TensorView &w);

} // namespace qllm
