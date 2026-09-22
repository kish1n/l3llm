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
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_mlp");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();

    // The residual stream as the MLP branch sees it: embedding plus the
    // attention block's output, both taken from the reference.
    std::vector<float> x_mid = qllm::testing::load_golden("embed_output");
    const std::vector<float> attn_out = qllm::testing::load_golden("layer0_attn_output");
    for (std::size_t i = 0; i < x_mid.size(); ++i) {
        x_mid[i] += attn_out[i];
    }

    const std::vector<float> h =
        qllm::rmsnorm(x_mid, fx.weights.at("model.layers.0.post_attention_layernorm.weight"),
                      fx.config.rms_norm_eps);
    std::println("[info ] hidden={} -> intermediate={} -> hidden={}", fx.config.hidden_size,
                 fx.config.intermediate_size, fx.config.hidden_size);

    const std::vector<float> out =
        qllm::mlp(h, fx.weights.at("model.layers.0.mlp.gate_proj.weight"),
                  fx.weights.at("model.layers.0.mlp.up_proj.weight"),
                  fx.weights.at("model.layers.0.mlp.down_proj.weight"));

    const auto result = qllm::testing::compare("layer0_mlp_output", out.data(), out.size(),
                                               /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
