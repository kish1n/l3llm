// Layer 0's Q and K after RoPE vs reference/golden/layer0_{q,k}_roped.bin.
//
// Two things are under test at once: the rotate_half pairing (j with
// j + head_dim/2, HF's convention, not the paper's adjacent pairs) and the
// llama3 frequency rescaling that stretches 8192 to 128k. Getting either
// wrong produces structured error, not noise.
//
// V has no golden counterpart here because RoPE is never applied to it.

#include <print>
#include <vector>

#include "compare.h"
#include "embed.h"
#include "fixture.h"
#include "linear.h"
#include "rmsnorm.h"
#include "rope.h"

int main() {
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_rope");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();
    const std::int64_t n_tokens = static_cast<std::int64_t>(fx.token_ids.size());

    const std::vector<float> x = qllm::embed(fx.weights, fx.config, fx.token_ids);
    const std::vector<float> h = qllm::rmsnorm(
        x, fx.weights.at("model.layers.0.input_layernorm.weight"), fx.config.rms_norm_eps);

    const qllm::RopeTable table = qllm::rope_table(fx.config, n_tokens);
    std::println("[info ] head_dim={} theta={:g} rope_type={}", fx.config.head_dim,
                 fx.config.rope_theta,
                 fx.config.rope_scaling ? fx.config.rope_scaling->rope_type : "none");

    bool ok = true;
    for (const auto &[golden, tensor, n_heads] :
         {std::tuple{"layer0_q_roped", "model.layers.0.self_attn.q_proj.weight",
                     fx.config.num_attention_heads},
          std::tuple{"layer0_k_roped", "model.layers.0.self_attn.k_proj.weight",
                     fx.config.num_key_value_heads}}) {
        const std::vector<float> proj = qllm::linear(h, fx.weights.at(tensor));
        std::vector<float> heads = qllm::split_heads(proj, n_heads, fx.config.head_dim);
        qllm::apply_rope(heads, n_heads, n_tokens, table);
        ok &= qllm::testing::compare(golden, heads.data(), heads.size(),
                                     /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, QLLM_GOLDEN_DIR)
                  .ok;
    }
    return ok ? 0 : 1;
}
