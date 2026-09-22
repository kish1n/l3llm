#pragma once

// SwiGLU feed-forward block:
//
//   y = down( silu(gate(h)) * up(h) )        silu(z) = z * sigmoid(z)
//
// Two parallel projections up to intermediate_size (8192 here, a 4x
// expansion), an elementwise gate between them, then one projection back
// down. Three matrices rather than the classic two is why the MLP holds
// 805M of this model's 1.24B parameters -- nearly 5x the attention blocks.

#include <span>
#include <vector>

#include "safetensors.h"

namespace qllm {

// h is [n_tokens, hidden] float32 (already normed). gate and up are
// [intermediate, hidden], down is [hidden, intermediate].
// Returns [n_tokens, hidden].
std::vector<float> mlp(std::span<const float> h, const TensorView &gate, const TensorView &up,
                       const TensorView &down);

} // namespace qllm
