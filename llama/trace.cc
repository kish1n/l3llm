#include "trace.h"

#include <format>
#include <fstream>
#include <iomanip>
#include <locale>

#include "die.h"

namespace qllm {
namespace {

// JSON quoting (std::quoted alone does not escape control characters).
std::string quote(std::string_view value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            out += std::format("\\u{:04x}", c);
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + '"';
}

} // namespace

thread_local TraceSession *TraceSession::active_ = nullptr;

TraceSession::TraceSession(bool enabled, bool detailed_attention)
    : enabled_(enabled), detailed_attention_(detailed_attention) {
    if (enabled_) {
        previous_ = active_;
        events_.reserve(4096);
        origin_ = Clock::now();
        active_ = this;
    }
}

TraceSession::~TraceSession() {
    if (enabled_) {
        active_ = previous_;
    }
}

std::size_t TraceSession::begin(std::string_view name, double flops, std::string_view context) {
    const std::size_t index = events_.size();
    events_.push_back({std::string(name), std::string(context), {}, 0, flops, current_});
    current_ = index;
    events_[index].start = Clock::now();
    return index;
}

void TraceSession::end(std::size_t index) {
    const auto end = Clock::now();
    Event &event = events_[index];
    event.duration_ns = std::chrono::duration<double, std::nano>(end - event.start).count();
    current_ = event.parent;
    if (current_ != no_parent) {
        events_[current_].flops += event.flops;
    }
}

void TraceSession::write(const std::filesystem::path &path) const {
    if (current_ != no_parent) {
        die("trace: cannot write while an operator region is open");
    }
    std::ofstream out(path);
    if (!out) {
        die("trace: could not open {}", path.string());
    }
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    out << R"({"displayTimeUnit":"ms","traceEvents":[)"
        << R"({"ph":"M","pid":1,"tid":1,"name":"process_name","args":{"name":"qllm"}},)"
        << R"({"ph":"M","pid":1,"tid":1,"name":"thread_name","args":{"name":"forward thread"}})";
    for (const Event &event : events_) {
        const double start_us =
            std::chrono::duration<double, std::micro>(event.start - origin_).count();
        // FLOPs/ns is numerically GFLOP/s. These are modeled arithmetic
        // operations, not a measurement of retired floating-point instructions.
        const bool has_rate = event.flops > 0 && event.duration_ns > 0;
        const double gflops = has_rate ? event.flops / event.duration_ns : 0;
        const std::string label =
            has_rate ? std::format("{} | {:.3f} GFLOP/s (est.)", event.name, gflops)
                     : std::format("{} | FLOP/s n/a", event.name);
        out << ",{\"ph\":\"X\",\"cat\":\"operator\",\"pid\":1,\"tid\":1,\"name\":" << quote(label)
            << ",\"ts\":" << start_us << ",\"dur\":" << event.duration_ns / 1000.0
            << ",\"args\":{\"operator\":" << quote(event.name)
            << ",\"context\":" << quote(event.context) << ",\"estimated_flops\":" << event.flops
            << ",\"duration_ms\":" << event.duration_ns / 1e6
            << ",\"estimated_gflops_per_second\":";
        if (has_rate) {
            out << gflops;
        } else {
            out << "null";
        }
        out << R"(,"timing":"inclusive wall time","flop_convention":"modeled add/sub/mul/div; FMA=2; excludes transcendental functions, conversions, comparisons and RoPE table setup"}})";
    }
    out << "]}\n";
    out.close();
    if (!out) {
        die("trace: failed writing {}", path.string());
    }
}

} // namespace qllm
