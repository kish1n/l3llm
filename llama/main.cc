// Usage: llama [--trace file.json] [--trace-attention] [--warmup N] [model-dir] [token-id ...]
//   Falls back to $L3LLM_MODEL_DIR when no model dir is given. With token
//   ids, runs a forward pass and prints the most likely next tokens;
//   without them, stops after the load summary. There is no tokenizer
//   yet, so ids come from the caller (reference/golden/manifest.json has
//   a prompt's worth).

#include <algorithm>
#include <charconv>
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
#include "trace.h"

// Owns the mmap: move-only, because SafeTensors is.
struct LoadedModel {
    l3llm::ModelConfig config;
    l3llm::SafeTensors weights;
};

LoadedModel load_model(std::string_view model_dir) {
    std::println("1. Reading Inputs");
    std::println("   model dir: {}", model_dir);

    l3llm::ModelConfig cfg = l3llm::ModelConfig::load(model_dir);
    l3llm::print_config(cfg);
    l3llm::SafeTensors weights =
        l3llm::SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors");
    l3llm::print_safetensors(weights, cfg);

    // Flat [vocab_size, hidden_size] in row-major order, so the first
    // hidden_size elements are token 0's embedding. Widening four of them
    // one at a time -- proof the mapping and the BF16 shift are wired up.
    // bf16() is a span into the mmap; nothing here copies the matrix.
    const auto token_embeddings = weights.at("model.embed_tokens.weight").bf16();
    std::println();
    std::println("   embed_tokens row 0, first 4: {:.6f} {:.6f} {:.6f} {:.6f}",
                 l3llm::bf16_to_f32(token_embeddings[0]), l3llm::bf16_to_f32(token_embeddings[1]),
                 l3llm::bf16_to_f32(token_embeddings[2]), l3llm::bf16_to_f32(token_embeddings[3]));
    return LoadedModel{std::move(cfg), std::move(weights)};
}

int main(int argc, char **argv) {
    const auto usage = [&] {
        std::println(
            "usage: {} [--trace file.json] [--trace-attention] [--warmup N] "
            "<model-dir> [token-id ...]\n"
            "  --trace             save the forward operator timeline for ui.perfetto.dev\n"
            "  --trace-attention   include per-row scores, softmax and weighted values\n"
            "  --warmup N          run N untraced forwards before the measured pass\n"
            "  --help              show this help\n"
            "  L3LLM_MODEL_DIR is used when no model directory is given",
            argv[0]);
    };
    std::string trace_path;
    bool detailed_attention = false;
    int warmup = 0;
    bool options = true;
    std::vector<std::string_view> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (options && arg == "--") {
            options = false;
        } else if (options && arg == "--help") {
            usage();
            return 0;
        } else if (options && arg == "--trace-attention") {
            detailed_attention = true;
        } else if (options && (arg == "--trace" || arg == "--warmup")) {
            if (++i == argc) {
                std::println(stderr, "{} requires a value", arg);
                return 2;
            }
            const std::string_view value = argv[i];
            if (arg == "--trace") {
                trace_path = value;
                if (trace_path.empty()) {
                    std::println(stderr, "--trace requires a nonempty file path");
                    return 2;
                }
            } else {
                const auto [end, error] =
                    std::from_chars(value.data(), value.data() + value.size(), warmup);
                if (error != std::errc{} || end != value.data() + value.size() || warmup < 0) {
                    std::println(stderr, "--warmup requires a nonnegative integer");
                    return 2;
                }
            }
        } else if (options && arg.starts_with("--")) {
            std::println(stderr, "unknown option: {}", arg);
            return 2;
        } else {
            positional.push_back(arg);
        }
    }
    if (detailed_attention && trace_path.empty()) {
        std::println(stderr, "--trace-attention requires --trace file.json");
        return 2;
    }

    std::string model_dir;
    if (!positional.empty()) {
        model_dir = positional.front();
    } else if (const char *env = std::getenv("L3LLM_MODEL_DIR")) {
        model_dir = env;
    } else {
        usage();
        return 2;
    }

    std::vector<std::int32_t> token_ids;
    for (std::size_t i = 1; i < positional.size(); ++i) {
        const std::string_view value = positional[i];
        std::int32_t id;
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), id);
        if (error != std::errc{} || end != value.data() + value.size() || id < 0) {
            std::println(stderr, "invalid token id: {}", value);
            return 2;
        }
        token_ids.push_back(id);
    }
    if (token_ids.empty() && (!trace_path.empty() || warmup > 0)) {
        std::println(stderr, "tracing and warmup require token ids after the model directory");
        return 2;
    }

    const LoadedModel model = load_model(model_dir);
    if (token_ids.empty()) {
        std::println();
        std::println("2. No token ids given; pass them after the model dir to run a forward pass.");
        return 0;
    }

    std::println();
    std::println("2. Forward pass");
    std::println("   tokens              {}", token_ids.size());

    for (int i = 0; i < warmup; ++i) {
        (void)l3llm::forward(model.weights, model.config, token_ids);
    }
    const l3llm::TraceSession trace(!trace_path.empty(), detailed_attention);
    const l3llm::ForwardResult out = l3llm::forward(model.weights, model.config, token_ids);
    if (!trace_path.empty()) {
        trace.write(trace_path);
        std::println("   operator timeline   {} (open in https://ui.perfetto.dev)", trace_path);
    }

    // Top 5 of the last row by logit; a partial sort beats sorting 128k.
    const std::size_t vocab = static_cast<std::size_t>(model.config.vocab_size);
    const float *row = out.logits.data() + (out.logits.size() - vocab);
    std::vector<std::int32_t> order(vocab);
    for (std::size_t i = 0; i < vocab; ++i) {
        order[i] = static_cast<std::int32_t>(i);
    }
    const std::size_t kTop = std::min<std::size_t>(5, vocab);
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                      [row](std::int32_t a, std::int32_t b) { return row[a] > row[b]; });

    std::println();
    std::println("   most likely next tokens");
    for (std::size_t i = 0; i < kTop; ++i) {
        std::println("     {:>6}  logit {:>9.4f}", order[i], row[order[i]]);
    }

    return 0;
}
