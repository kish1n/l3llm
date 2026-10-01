// Layer 0's SwiGLU block vs reference/golden/layer0_mlp_output.bin.
//
// Fed from goldens rather than from a recomputed chain: the MLP's input is
// rmsnorm(embed + attn_out), and both of those are already dumped. That
// isolates this test from upstream error, so a failure is the activation,
// the gate, or a swapped gate/up matrix -- nothing earlier.

#include <cstddef>
#include <print>
#include <vector>

#include "compare.h"
#include "fixture.h"
#include "mlp.h"
#include "rmsnorm.h"

int main() {
    if (!l3llm::testing::have_model()) {
        l3llm::testing::print_skip("test_mlp");
        return l3llm::testing::kSkip;
    }
    const l3llm::testing::Fixture fx = l3llm::testing::load();

    // The residual stream as the MLP branch sees it: embedding plus the
    // attention block's output, both taken from the reference.
    std::vector<float> x_mid = l3llm::testing::load_golden("embed_output");
    const std::vector<float> attn_out = l3llm::testing::load_golden("layer0_attn_output");
    for (std::size_t i = 0; i < x_mid.size(); ++i) {
        x_mid[i] += attn_out[i];
    }

    const std::vector<float> h =
        l3llm::rmsnorm(x_mid, fx.weights.at("model.layers.0.post_attention_layernorm.weight"),
                      fx.config.rms_norm_eps);
    std::println("[info ] hidden={} -> intermediate={} -> hidden={}", fx.config.hidden_size,
                 fx.config.intermediate_size, fx.config.hidden_size);

    const std::vector<float> out =
        l3llm::mlp(h, fx.weights.at("model.layers.0.mlp.gate_proj.weight"),
                  fx.weights.at("model.layers.0.mlp.up_proj.weight"),
                  fx.weights.at("model.layers.0.mlp.down_proj.weight"));

    const auto result = l3llm::testing::compare("layer0_mlp_output", out.data(), out.size(),
                                               /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, L3LLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
