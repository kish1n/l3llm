#pragma once

// Shared setup for the golden tests: finding the weights, and reading the
// prompt's token ids from the manifest so the tests follow whatever
// dump_golden.py was last run with.
//
// Every test needs the real weights, so when QLLM_MODEL_DIR is unset they
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

namespace qllm::testing {

inline constexpr int kSkip = 77; // matches SKIP_RETURN_CODE in CMakeLists.txt

struct Fixture {
    ModelConfig config;
    SafeTensors weights;
    std::vector<std::int32_t> token_ids;
};

// The prompt's token ids, from reference/golden/manifest.json.
inline std::vector<std::int32_t> golden_token_ids() {
    const std::filesystem::path path = std::filesystem::path(QLLM_GOLDEN_DIR) / "manifest.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        die("fixture: could not open {}", path.string());
    }
    std::ostringstream buf;
    buf << f.rdbuf();

    const json::Value root = json::parse(buf.str(), path.string());
    const json::Value *ids = root.find("token_ids");
    if (ids == nullptr) {
        die("fixture: {} has no token_ids", path.string());
    }

    std::vector<std::int32_t> out;
    for (const json::Value &v : ids->as_array("token_ids")) {
        out.push_back(static_cast<std::int32_t>(v.as_int64("token_ids[]")));
    }
    return out;
}

// Returns nullopt when QLLM_MODEL_DIR is unset; the caller returns kSkip.
inline bool have_model() { return std::getenv("QLLM_MODEL_DIR") != nullptr; }

inline void print_skip(std::string_view test) {
    std::println("[skip ] {}: set QLLM_MODEL_DIR to a directory with config.json "
                 "and model.safetensors",
                 test);
}

inline Fixture load() {
    const char *model_dir = std::getenv("QLLM_MODEL_DIR");
    if (model_dir == nullptr) {
        die("fixture: QLLM_MODEL_DIR is unset (call have_model() first)");
    }
    return Fixture{
        ModelConfig::load(model_dir),
        SafeTensors::open(std::filesystem::path(model_dir) / "model.safetensors"),
        golden_token_ids(),
    };
}

} // namespace qllm::testing
