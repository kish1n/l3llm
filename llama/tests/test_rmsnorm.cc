// Layer 0's input_layernorm vs reference/golden/layer0_attn_norm_output.bin.
//
// First step with real arithmetic, so unlike test_embed this cannot be
// bit-exact: the sum of 2048 squares accumulates in a different order than
// PyTorch's reduction. Tolerances are tight enough that a wrong formula
// (mean subtraction, gain applied before the scale, eps inside vs outside
// the sqrt) still fails loudly.

#include <print>
#include <vector>

#include "compare.h"
#include "embed.h"
#include "fixture.h"
#include "rmsnorm.h"

int main() {
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_rmsnorm");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();

    const std::vector<float> x = qllm::embed(fx.weights, fx.config, fx.token_ids);
    const qllm::TensorView &gain = fx.weights.at("model.layers.0.input_layernorm.weight");
    const std::vector<float> out = qllm::rmsnorm(x, gain, fx.config.rms_norm_eps);

    std::println("[info ] {} tokens x {} hidden, eps={:g}", fx.token_ids.size(),
                 fx.config.hidden_size, fx.config.rms_norm_eps);

    const auto result = qllm::testing::compare("layer0_attn_norm_output", out.data(), out.size(),
                                               /*abs_tol=*/1e-5, /*rel_tol=*/1e-5, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
