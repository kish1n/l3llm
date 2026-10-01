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
    if (!l3llm::testing::have_model()) {
        l3llm::testing::print_skip("test_rmsnorm");
        return l3llm::testing::kSkip;
    }
    const l3llm::testing::Fixture fx = l3llm::testing::load();

    const std::vector<float> x = l3llm::embed(fx.weights, fx.config, fx.token_ids);
    const l3llm::TensorView &gain = fx.weights.at("model.layers.0.input_layernorm.weight");
    const std::vector<float> out = l3llm::rmsnorm(x, gain, fx.config.rms_norm_eps);

    std::println("[info ] {} tokens x {} hidden, eps={:g}", fx.token_ids.size(),
                 fx.config.hidden_size, fx.config.rms_norm_eps);

    const auto result = l3llm::testing::compare("layer0_attn_norm_output", out.data(), out.size(),
                                               /*abs_tol=*/1e-5, /*rel_tol=*/1e-5, L3LLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
