// Layer 0's q/k/v projections vs reference/golden/layer0_{q,k,v}_proj.bin.
//
// First matmul, so this is the first test whose error is dominated by
// summation order rather than by a single rounding: a 2048-term dot
// product accumulated in fp32 drifts from the reference's blocked BLAS
// reduction by roughly sqrt(2048) * eps.
//
// The k/v outputs are 4x narrower than q -- that is GQA in the weights:
// 8 KV heads x 64 against 32 query heads x 64.

#include <print>
#include <vector>

#include "compare.h"
#include "embed.h"
#include "fixture.h"
#include "linear.h"
#include "rmsnorm.h"

int main() {
    if (!l3llm::testing::have_model()) {
        l3llm::testing::print_skip("test_qkv");
        return l3llm::testing::kSkip;
    }
    const l3llm::testing::Fixture fx = l3llm::testing::load();

    const std::vector<float> x = l3llm::embed(fx.weights, fx.config, fx.token_ids);
    const std::vector<float> h = l3llm::rmsnorm(
        x, fx.weights.at("model.layers.0.input_layernorm.weight"), fx.config.rms_norm_eps);

    bool ok = true;
    for (const auto &[golden, tensor] :
         {std::pair{"layer0_q_proj", "model.layers.0.self_attn.q_proj.weight"},
          std::pair{"layer0_k_proj", "model.layers.0.self_attn.k_proj.weight"},
          std::pair{"layer0_v_proj", "model.layers.0.self_attn.v_proj.weight"}}) {
        const std::vector<float> y = l3llm::linear(h, fx.weights.at(tensor));
        ok &= l3llm::testing::compare(golden, y.data(), y.size(),
                                     /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, L3LLM_GOLDEN_DIR)
                  .ok;
    }
    return ok ? 0 : 1;
}
