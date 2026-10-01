// The full forward pass against every golden the reference dumps.
//
// Checks all 16 layer outputs as they go by, then the final norm, then the
// logits -- so a divergence is localized to the layer where it first
// exceeds tolerance rather than only showing up at the end.
//
// The last assertion is the one that matters end to end: greedy argmax
// over the final row must reproduce the first token the reference
// generated. That is an integer, so it either matches or it does not --
// no tolerance involved.

#include <cstdint>
#include <format>
#include <print>
#include <vector>

#include "compare.h"
#include "fixture.h"
#include "model.h"

int main() {
    if (!l3llm::testing::have_model()) {
        l3llm::testing::print_skip("test_model");
        return l3llm::testing::kSkip;
    }
    const l3llm::testing::Fixture fx = l3llm::testing::load();
    std::println("[info ] {} layers, {} tokens, vocab {}", fx.config.num_hidden_layers,
                 fx.token_ids.size(), fx.config.vocab_size);

    bool ok = true;
    const auto on_layer = [&](std::int64_t index, std::span<const float> x) {
        ok &= l3llm::testing::compare(std::format("layer{}_output", index), x.data(), x.size(),
                                     /*abs_tol=*/1e-3, /*rel_tol=*/1e-3, L3LLM_GOLDEN_DIR)
                  .ok;
    };

    const l3llm::ForwardResult out = l3llm::forward(fx.weights, fx.config, fx.token_ids, on_layer);

    ok &= l3llm::testing::compare("final_norm_output", out.hidden.data(), out.hidden.size(),
                                 /*abs_tol=*/1e-3, /*rel_tol=*/1e-3, L3LLM_GOLDEN_DIR)
              .ok;
    ok &= l3llm::testing::compare("logits", out.logits.data(), out.logits.size(),
                                 /*abs_tol=*/1e-2, /*rel_tol=*/1e-2, L3LLM_GOLDEN_DIR)
              .ok;

    // Greedy next token: exact integer match or bust.
    const std::int32_t predicted = l3llm::argmax_last(out, fx.config);
    const std::int32_t expected = l3llm::testing::golden_ints("generated_new_tokens").at(0);
    std::println("[check] next token: got {}, reference {}  {}", predicted, expected,
                 predicted == expected ? "OK" : "FAIL");
    ok &= (predicted == expected);

    return ok ? 0 : 1;
}
