// A whole decoder layer vs reference/golden/layer0_output.bin.
//
// Unlike test_mlp this recomputes everything from the token ids, so it is
// the first end-to-end check: embed, both norms, all seven projections,
// RoPE, causal GQA, SwiGLU, and -- the part nothing else covers -- the two
// residual adds reading the pre-norm x.

#include <cstdint>
#include <print>
#include <vector>

#include "compare.h"
#include "embed.h"
#include "fixture.h"
#include "layer.h"
#include "rope.h"

int main() {
    if (!qllm::testing::have_model()) {
        qllm::testing::print_skip("test_layer");
        return qllm::testing::kSkip;
    }
    const qllm::testing::Fixture fx = qllm::testing::load();
    const std::int64_t n_tokens = static_cast<std::int64_t>(fx.token_ids.size());

    std::vector<float> x = qllm::embed(fx.weights, fx.config, fx.token_ids);
    const qllm::RopeTable table = qllm::rope_table(fx.config, n_tokens);

    qllm::decoder_layer(x, fx.weights, fx.config, /*index=*/0, table, n_tokens);
    std::println("[info ] layer 0 applied in place to {} x {}", n_tokens, fx.config.hidden_size);

    const auto result = qllm::testing::compare("layer0_output", x.data(), x.size(),
                                               /*abs_tol=*/1e-4, /*rel_tol=*/1e-4, QLLM_GOLDEN_DIR);
    return result.ok ? 0 : 1;
}
