#pragma once

// Rotary position embedding.
//
// Position is not added to the residual stream -- there is no learned
// position table in the file. Instead, inside every layer, Q and K are
// rotated by an angle proportional to the token's position. V is left
// alone: position decides *which* tokens attend to which, not what gets
// retrieved.
//
// Within a head's head_dim values, dimension j pairs with j + head_dim/2
// and the pair is rotated by m * inv_freq[j] for a token at position m.
// That j <-> j+half pairing is Hugging Face's "rotate_half" convention;
// the original paper pairs adjacent dimensions (2j, 2j+1) instead. The two
// give different numbers, and the golden dump comes from HF.
//
// The dot product of a rotated query at position m with a rotated key at
// position n depends only on m - n: absolute rotations, relative effect.

#include <cstdint>
#include <span>
#include <vector>

#include "config.h"

namespace qllm {

// cos/sin for every (position, frequency) pair, each [n_positions,
// head_dim/2]. Built once and shared by all layers -- the angles depend
// only on the config, not on the weights.
struct RopeTable {
    std::int64_t head_dim = 0;
    std::int64_t n_positions = 0;
    std::vector<float> cos; // [n_positions, head_dim/2]
    std::vector<float> sin;
};

// Applies cfg.rope_theta and, when present, the "llama3" frequency
// rescaling that stretches the 8192-token training context to 128k.
// die()s on an unrecognized rope_type.
RopeTable rope_table(const ModelConfig &cfg, std::int64_t n_positions);

// [n_tokens, n_heads * head_dim] -> [n_heads, n_tokens, head_dim].
// Attention works per head, so the head index becomes the outer stride.
std::vector<float> split_heads(std::span<const float> x, std::int64_t n_heads,
                               std::int64_t head_dim);

// Rotates in place over a [n_heads, n_tokens, head_dim] buffer, treating
// token t as position t.
void apply_rope(std::span<float> x, std::int64_t n_heads, std::int64_t n_tokens,
                const RopeTable &table);

} // namespace qllm
