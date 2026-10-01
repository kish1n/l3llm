#include "linear.h"

#include <cstddef>
#include <cstdint>

#include "die.h"
#include "trace.h"

namespace l3llm {

std::vector<float> linear(std::span<const float> x, const TensorView &w) {
    if (w.shape.size() != 2) {
        die("linear: weight '{}' has rank {}, expected 2", w.name, w.shape.size());
    }
    const std::size_t out_features = static_cast<std::size_t>(w.shape[0]);
    const std::size_t in_features = static_cast<std::size_t>(w.shape[1]);
    if (in_features == 0 || x.size() % in_features != 0) {
        die("linear: {} floats is not a whole number of rows of width {} (weight '{}')", x.size(),
            in_features, w.name);
    }

    const std::span<const std::uint16_t> wq = w.bf16();
    const std::size_t n_tokens = x.size() / in_features;

    // Use the projection's role for the bar and retain its full tensor name.
    std::string_view role = w.name;
    if (role.ends_with(".weight")) {
        role.remove_suffix(7);
    }
    if (const auto dot = role.rfind('.'); dot != std::string_view::npos) {
        role.remove_prefix(dot + 1);
    }
    const TraceScope trace(role, 2.0 * n_tokens * out_features * in_features, w.name);

    std::vector<float> out(n_tokens * out_features);
    for (std::size_t t = 0; t < n_tokens; ++t) {
        const float *row = x.data() + t * in_features;
        float *dst = out.data() + t * out_features;
        for (std::size_t o = 0; o < out_features; ++o) {
            const std::uint16_t *wrow = wq.data() + o * in_features;
            // Naive dot product: correct first, blocked later. Both operands
            // walk contiguous memory, so this is the shape a vectorized
            // kernel will keep -- only the loop nest around it changes.
            float acc = 0.0f;
            for (std::size_t i = 0; i < in_features; ++i) {
                acc += row[i] * bf16_to_f32(wrow[i]);
            }
            dst[o] = acc;
        }
    }
    return out;
}

} // namespace l3llm
