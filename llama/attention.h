#pragma once

// Causal grouped-query attention.
//
//   scores = q . k^T / sqrt(head_dim), masked so token i sees only j <= i
//   out    = softmax(scores) . v
//
// Grouped-query: there are 32 query heads but only 8 KV heads, so each KV
// head is shared by heads_per_kv_group() = 4 query heads. Nothing is
// materialized for that sharing -- query head h simply reads KV head
// h / group. (Hugging Face's repeat_kv() tiles the KV tensors instead;
// same arithmetic, 4x the memory.)
//
// The mask is structural rather than additive: the j loop stops at i, so
// no -inf ever enters the softmax.

#include <cstdint>
#include <span>
#include <vector>

#include "config.h"

namespace l3llm {

// q is [n_heads, n_tokens, head_dim], k and v are [n_kv_heads, n_tokens,
// head_dim], all post-RoPE except v. Returns [n_heads, n_tokens, head_dim].
std::vector<float> attention(std::span<const float> q, std::span<const float> k,
                             std::span<const float> v, const ModelConfig &cfg,
                             std::int64_t n_tokens);

// [n_heads, n_tokens, head_dim] -> [n_tokens, n_heads * head_dim], the
// inverse of split_heads(). o_proj expects the token-major layout back.
std::vector<float> merge_heads(std::span<const float> x, std::int64_t n_heads,
                               std::int64_t n_tokens, std::int64_t head_dim);

} // namespace l3llm
