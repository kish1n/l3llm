// Token embedding gather vs reference/golden/embed_output.bin.
//
// Expects a bit-exact match: BF16 -> F32 widening is lossless and the
// reference runs the same lookup in fp32, so any nonzero error means a
// wrong row, a wrong stride, or a bad mapping -- never rounding. Hence
// the zero tolerances.

#include <print>
#include <vector>

#include "compare.h"
#include "embed.h"
#include "fixture.h"

int main() {
    if (!l3llm::testing::have_model()) {
        l3llm::testing::print_skip("test_embed");
        return l3llm::testing::kSkip;
    }
    const l3llm::testing::Fixture fx = l3llm::testing::load();

    const std::vector<float> out = l3llm::embed(fx.weights, fx.config, fx.token_ids);
    std::println("[info ] {} tokens x {} hidden = {} floats", fx.token_ids.size(),
                 fx.config.hidden_size, out.size());

    const auto result = l3llm::testing::compare("embed_output", out.data(), out.size(),
                                               /*abs_tol=*/0.0, /*rel_tol=*/0.0, L3LLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
