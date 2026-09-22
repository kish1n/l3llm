// Layer 0's full attention block vs reference/golden/layer0_attn_output.bin.
//
// The golden is the self_attn module's return value, so it includes
// o_proj but not the residual add. Chain under test:
//
//   embed -> rmsnorm -> q/k/v proj -> split_heads -> RoPE (q, k only)
//         -> causal GQA -> merge_heads -> o_proj
//
// Everything before attention is already pinned by its own test, so a
// failure here localizes to the masking, the softmax, the 1/sqrt(64)
// scale, or the query-head-to-KV-head mapping.

#include <cstdint>
#include <print>
#include <vector>

#include "attention.h"
#include "compare.h"
#include "embed.h"
#include "fixture.h"
#include "linear.h"
#include "rmsnorm.h"
#include "rope.h"

int main() {
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_attention");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();
    const qllm::ModelConfig &cfg = fx.config;
    const std::int64_t n_tokens = static_cast<std::int64_t>(fx.token_ids.size());

    const std::vector<float> x = qllm::embed(fx.weights, cfg, fx.token_ids);
    const std::vector<float> h =
        qllm::rmsnorm(x, fx.weights.at("model.layers.0.input_layernorm.weight"), cfg.rms_norm_eps);

    const qllm::RopeTable table = qllm::rope_table(cfg, n_tokens);

    std::vector<float> q =
        qllm::split_heads(qllm::linear(h, fx.weights.at("model.layers.0.self_attn.q_proj.weight")),
                          cfg.num_attention_heads, cfg.head_dim);
    std::vector<float> k =
        qllm::split_heads(qllm::linear(h, fx.weights.at("model.layers.0.self_attn.k_proj.weight")),
                          cfg.num_key_value_heads, cfg.head_dim);
    // V is split into heads but never rotated.
    const std::vector<float> v =
        qllm::split_heads(qllm::linear(h, fx.weights.at("model.layers.0.self_attn.v_proj.weight")),
                          cfg.num_key_value_heads, cfg.head_dim);

    qllm::apply_rope(q, cfg.num_attention_heads, n_tokens, table);
    qllm::apply_rope(k, cfg.num_key_value_heads, n_tokens, table);

    std::println("[info ] {} q heads / {} kv heads ({}:1 GQA), head_dim={}, {} tokens",
                 cfg.num_attention_heads, cfg.num_key_value_heads, cfg.heads_per_kv_group(),
                 cfg.head_dim, n_tokens);

    const std::vector<float> ctx = qllm::attention(q, k, v, cfg, n_tokens);
    const std::vector<float> merged =
        qllm::merge_heads(ctx, cfg.num_attention_heads, n_tokens, cfg.head_dim);
    const std::vector<float> out =
        qllm::linear(merged, fx.weights.at("model.layers.0.self_attn.o_proj.weight"));

    const auto result = qllm::testing::compare("layer0_attn_output", out.data(), out.size(),
                                               /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
