// Usage: llama [model-dir] [token-id ...]
//   Falls back to $QLLM_MODEL_DIR when no model dir is given. With token
//   ids, runs a forward pass and prints the most likely next tokens;
//   without them, stops after the load summary. There is no tokenizer
//   yet, so ids come from the caller (reference/golden/manifest.json has
//   a prompt's worth).

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"
#include "model.h"
#include "safetensors.h"
#include "summary.h"

// Owns the mmap: move-only, because SafeTensors is.
struct LoadedModel {
    qllm::ModelConfig config;
    qllm::SafeTensors weights;
};

LoadedModel load_model(std::string_view model_dir) {
    std::println("1. Reading Inputs");
    std::println("   model dir: {}", model_dir);

    qllm::ModelConfig cfg = qllm::ModelConfig::load(model_dir);
    qllm::print_config(cfg);
    qllm::SafeTensors weights =
        qllm::SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors");
    qllm::print_safetensors(weights, cfg);

    // Flat [vocab_size, hidden_size] in row-major order, so the first
    // hidden_size elements are token 0's embedding. Widening four of them
    // one at a time -- proof the mapping and the BF16 shift are wired up.
    // bf16() is a span into the mmap; nothing here copies the matrix.
    const auto token_embeddings = weights.at("model.embed_tokens.weight").bf16();
    std::println();
    std::println("   embed_tokens row 0, first 4: {:.6f} {:.6f} {:.6f} {:.6f}",
                 qllm::bf16_to_f32(token_embeddings[0]), qllm::bf16_to_f32(token_embeddings[1]),
                 qllm::bf16_to_f32(token_embeddings[2]), qllm::bf16_to_f32(token_embeddings[3]));
    return LoadedModel{std::move(cfg), std::move(weights)};
}

int main(int argc, char **argv) {
    std::string model_dir;
    if (argc > 1) {
        model_dir = argv[1];
    } else if (const char *env = std::getenv("QLLM_MODEL_DIR")) {
        model_dir = env;
    } else {
        std::println(stderr,
                     "usage: {} <model-dir>\n"
                     "  or set QLLM_MODEL_DIR to a directory containing\n"
                     "  config.json and model.safetensors",
                     argv[0]);
        return 2;
    }

    const LoadedModel model = load_model(model_dir);

    std::vector<std::int32_t> token_ids;
    for (int i = 2; i < argc; ++i) {
        token_ids.push_back(static_cast<std::int32_t>(std::atoi(argv[i])));
    }
    if (token_ids.empty()) {
        std::println();
        std::println("2. No token ids given; pass them after the model dir to run a forward pass.");
        return 0;
    }

    std::println();
    std::println("2. Forward pass");
    std::println("   tokens              {}", token_ids.size());

    const qllm::ForwardResult out = qllm::forward(model.weights, model.config, token_ids);

    // Top 5 of the last row by logit; a partial sort beats sorting 128k.
    const std::size_t vocab = static_cast<std::size_t>(model.config.vocab_size);
    const float *row = out.logits.data() + (out.logits.size() - vocab);
    std::vector<std::int32_t> order(vocab);
    for (std::size_t i = 0; i < vocab; ++i) {
        order[i] = static_cast<std::int32_t>(i);
    }
    constexpr std::size_t kTop = 5;
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                      [row](std::int32_t a, std::int32_t b) { return row[a] > row[b]; });

    std::println();
    std::println("   most likely next tokens");
    for (std::size_t i = 0; i < kTop; ++i) {
        std::println("     {:>6}  logit {:>9.4f}", order[i], row[order[i]]);
    }

    return 0;
}
