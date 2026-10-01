#pragma once

// Shared setup for the golden tests: finding the weights, and reading the
// prompt's token ids from the manifest so the tests follow whatever
// dump_golden.py was last run with.
//
// Every test needs the real weights, so when L3LLM_MODEL_DIR is unset they
// exit kSkip and CTest reports "Skipped" rather than a failure -- a fresh
// clone without a 2.3 GiB download still gets a green run.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "die.h"
#include "golden_dir_config.h"
#include "json.h"
#include "safetensors.h"

namespace l3llm::testing {

inline constexpr int kSkip = 77; // matches SKIP_RETURN_CODE in CMakeLists.txt

struct Fixture {
    ModelConfig config;
    SafeTensors weights;
    std::vector<std::int32_t> token_ids;
};

// An integer array from reference/golden/manifest.json -- "token_ids" for
// the prompt, "generated_new_tokens" for what greedy decoding produced.
inline std::vector<std::int32_t> golden_ints(std::string_view key) {
    const std::filesystem::path path = std::filesystem::path(L3LLM_GOLDEN_DIR) / "manifest.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        die("fixture: could not open {}", path.string());
    }
    std::ostringstream buf;
    buf << f.rdbuf();

    const json::Value root = json::parse(buf.str(), path.string());
    const json::Value *ids = root.find(key);
    if (ids == nullptr) {
        die("fixture: {} has no {}", path.string(), key);
    }

    std::vector<std::int32_t> out;
    for (const json::Value &v : ids->as_array(key)) {
        out.push_back(static_cast<std::int32_t>(v.as_int64(key)));
    }
    return out;
}

inline std::vector<std::int32_t> golden_token_ids() { return golden_ints("token_ids"); }

// A golden dump read back as float32, for tests that want to feed one
// step's reference output into the next step instead of recomputing the
// whole chain -- it isolates the step under test from upstream error.
inline std::vector<float> load_golden(std::string_view name) {
    const std::filesystem::path path =
        std::filesystem::path(L3LLM_GOLDEN_DIR) / (std::string(name) + ".bin");
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        die("fixture: could not open {}", path.string());
    }
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        die("fixture: {} is {} bytes, not a whole number of float32s", path.string(),
            static_cast<long long>(bytes));
    }
    std::vector<float> out(static_cast<std::size_t>(bytes) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), bytes);
    if (!f) {
        die("fixture: short read from {}", path.string());
    }
    return out;
}

// Returns nullopt when L3LLM_MODEL_DIR is unset; the caller returns kSkip.
inline bool have_model() { return std::getenv("L3LLM_MODEL_DIR") != nullptr; }

inline void print_skip(std::string_view test) {
    std::println("[skip ] {}: set L3LLM_MODEL_DIR to a directory with config.json "
                 "and model.safetensors",
                 test);
}

inline Fixture load() {
    const char *model_dir = std::getenv("L3LLM_MODEL_DIR");
    if (model_dir == nullptr) {
        die("fixture: L3LLM_MODEL_DIR is unset (call have_model() first)");
    }
    return Fixture{
        ModelConfig::load(model_dir),
        SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors"),
        golden_token_ids(),
    };
}

} // namespace l3llm::testing
