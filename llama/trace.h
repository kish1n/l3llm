#pragma once

// Optional, calling-thread-only operator tracing. No clocks or event allocations
// when disabled. Events stay in memory until write(), outside the measured pass.
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace qllm {

class TraceScope;

class TraceSession {
  public:
    explicit TraceSession(bool enabled = true, bool detailed_attention = false);
    ~TraceSession();
    TraceSession(const TraceSession &) = delete;
    TraceSession &operator=(const TraceSession &) = delete;

    void write(const std::filesystem::path &path) const;
    static bool enabled() { return active_ != nullptr; }
    static bool detailed_attention() { return active_ && active_->detailed_attention_; }

  private:
    friend class TraceScope;
    using Clock = std::chrono::steady_clock;
    static constexpr std::size_t no_parent = static_cast<std::size_t>(-1);
    struct Event {
        std::string name;
        std::string context;
        Clock::time_point start;
        double duration_ns = 0;
        double flops = 0;
        std::size_t parent = no_parent;
    };
    static thread_local TraceSession *active_;
    TraceSession *previous_ = nullptr;
    bool enabled_;
    bool detailed_attention_;
    Clock::time_point origin_;
    std::vector<Event> events_;
    std::size_t current_ = no_parent;

    std::size_t begin(std::string_view name, double flops, std::string_view context);
    void end(std::size_t index);
};

class TraceScope {
  public:
    explicit TraceScope(std::string_view name, double flops = 0, std::string_view context = {},
                        bool enabled = true)
        : session_(enabled ? TraceSession::active_ : nullptr) {
        if (session_) {
            index_ = session_->begin(name, flops, context);
        }
    }
    ~TraceScope() {
        if (session_) {
            session_->end(index_);
        }
    }
    TraceScope(const TraceScope &) = delete;
    TraceScope &operator=(const TraceScope &) = delete;

  private:
    TraceSession *session_;
    std::size_t index_ = 0;
};

} // namespace qllm
