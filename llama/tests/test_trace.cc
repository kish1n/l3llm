// No model download required: exercise exported counts and timing using small
// real kernels, and ensure tracing leaves their numerical output unchanged.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention.h"
#include "json.h"
#include "linear.h"
#include "trace.h"

namespace {

using qllm::json::Value;

void require(bool ok, std::string_view message) {
    if (!ok) {
        throw std::runtime_error(std::string(message));
    }
}

const Value &field(const Value &value, std::string_view key) {
    const Value *result = value.find(key);
    require(result != nullptr, key);
    return *result;
}

std::vector<Value> read_events(const qllm::TraceSession &trace) {
    const auto path = std::filesystem::path("test_trace_output.json");
    trace.write(path);
    std::ifstream input(path);
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    input.close();
    std::filesystem::remove(path);
    const Value root = qllm::json::parse(text, "trace test");
    std::vector<Value> events;
    for (const auto &event : field(root, "traceEvents").array) {
        if (field(event, "ph").string == "X") {
            events.push_back(event);
        }
    }
    return events;
}

double count(const Value &event) { return field(field(event, "args"), "estimated_flops").number; }

void test_nested_scopes() {
    const std::string unusual_name = "parent\"\\\n\t";
    const qllm::TraceSession trace;
    {
        const qllm::TraceScope parent(unusual_name, 10);
        {
            const qllm::TraceScope child("child", 20);
            const qllm::TraceScope grandchild("grandchild", 30);
        }
        const qllm::TraceScope copy("copy");
    }
    const auto events = read_events(trace);
    require(events.size() == 4, "scope count");
    require(count(events[0]) == 60 && count(events[1]) == 50 && count(events[2]) == 30,
            "inclusive counts must not double count grandchildren");
    require(field(field(events[0], "args"), "operator").string == unusual_name,
            "JSON string escaping");
    require(field(field(events[3], "args"), "estimated_gflops_per_second").is_null(),
            "copy has no FLOP rate");
    const double parent_start = field(events[0], "ts").number;
    const double parent_end = parent_start + field(events[0], "dur").number;
    for (const auto &event : events) {
        const double start = field(event, "ts").number;
        const double duration = field(event, "dur").number;
        require(start >= parent_start && start + duration <= parent_end + 1e-6,
                "children must fit inside parent time interval");
        if (count(event) > 0) {
            const double rate = field(field(event, "args"), "estimated_gflops_per_second").number;
            require(duration > 0 && std::isfinite(rate) &&
                        std::abs(rate * duration * 1000 - count(event)) < 1e-6,
                    "GFLOP/s conversion");
            require(field(event, "name").string.find("GFLOP/s") != std::string::npos,
                    "hover label includes rate");
        }
    }
}

void test_linear() {
    const std::vector<std::uint16_t> weights(12, 0x3f80); // 3 x 4, all 1.0 BF16
    const qllm::TensorView w{"model.layers.0.mlp.gate_proj.weight",
                             qllm::DType::BF16,
                             {3, 4},
                             std::as_bytes(std::span(weights))};
    const std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8}; // two tokens
    const auto baseline = qllm::linear(x, w);
    const qllm::TraceSession trace;
    require(qllm::linear(x, w) == baseline, "tracing changes linear output");
    const auto events = read_events(trace);
    require(events.size() == 1 && count(events[0]) == 48, "linear FLOPs: 2*2*3*4");
    require(field(field(events[0], "args"), "operator").string == "gate_proj", "projection role");
    require(field(field(events[0], "args"), "context").string == w.name, "full weight context");
}

void test_attention() {
    qllm::ModelConfig cfg;
    cfg.head_dim = 2;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    const std::vector<float> q{1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
    const std::vector<float> k{1, 0, 0, 1, 1, 1};
    const std::vector<float> v{2, 3, 5, 7, 11, 13};
    const auto baseline = qllm::attention(q, k, v, cfg, 3);
    for (const bool detailed : {false, true}) {
        const qllm::TraceSession trace(true, detailed);
        require(qllm::attention(q, k, v, cfg, 3) == baseline, "tracing changes attention output");
        const auto events = read_events(trace);
        require(events.size() == (detailed ? 19 : 1), "attention detail scope count");
        // Six causal pairs/head: 120 score+value ops, 30 softmax ops,
        // plus one shared scale division. Both modes must count 151.
        require(count(events[0]) == 151, "causal GQA FLOPs / no detail double counting");
        double softmax_flops = 0;
        for (const auto &event : events) {
            if (field(field(event, "args"), "operator").string == "softmax") {
                softmax_flops += count(event);
            }
        }
        require(softmax_flops == (detailed ? 30 : 0), "softmax FLOPs");
    }
}

void test_disabled_and_growth() {
    {
        const qllm::TraceSession trace(false);
        const qllm::TraceScope ignored("disabled", 42);
        require(!qllm::TraceSession::enabled(), "disabled session became active");
        require(read_events(trace).empty(), "disabled trace records events");
    }
    const qllm::TraceSession trace;
    {
        const qllm::TraceScope parent("parent");
        for (int i = 0; i < 4200; ++i) {
            const qllm::TraceScope child("child", 1);
        }
    }
    const auto events = read_events(trace);
    require(events.size() == 4201 && count(events[0]) == 4200,
            "growing event buffer invalidates nesting");
}

} // namespace

int main() {
    try {
        test_nested_scopes();
        test_linear();
        test_attention();
        test_disabled_and_growth();
        require(!qllm::TraceSession::enabled(), "session did not restore inactive state");
        std::println("[pass] trace counts, nesting, rates, escaping and kernel equivalence");
        return 0;
    } catch (const std::exception &error) {
        std::println(stderr, "[fail] {}", error.what());
        return 1;
    }
}
