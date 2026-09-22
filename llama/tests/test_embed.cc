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
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_embed");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();

    const std::vector<float> out = qllm::embed(fx.weights, fx.config, fx.token_ids);
    std::println("[info ] {} tokens x {} hidden = {} floats", fx.token_ids.size(),
                 fx.config.hidden_size, out.size());

    const auto result = qllm::testing::compare("embed_output", out.data(), out.size(),
                                               /*abs_tol=*/0.0, /*rel_tol=*/0.0, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
