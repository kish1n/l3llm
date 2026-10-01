#include "rmsnorm.h"

#include <cmath>
#include <cstdint>

#include "die.h"
#include "trace.h"

namespace l3llm {

std::vector<float> rmsnorm(std::span<const float> x, const TensorView &gamma, double eps) {
    if (gamma.shape.size() != 1) {
        die("rmsnorm: gain '{}' has rank {}, expected 1", gamma.name, gamma.shape.size());
    }
    const std::size_t hidden = static_cast<std::size_t>(gamma.shape[0]);
    if (hidden == 0 || x.size() % hidden != 0) {
        die("rmsnorm: {} floats is not a whole number of rows of width {}", x.size(), hidden);
    }

    const std::span<const std::uint16_t> g = gamma.bf16();
    const std::size_t n_tokens = x.size() / hidden;
    // Square + accumulate + two scaling multiplies per element; two
    // divisions and epsilon addition per row. sqrt is excluded.
    const TraceScope trace("rmsnorm", 4.0 * x.size() + 3.0 * n_tokens, gamma.name);

    std::vector<float> out(x.size());
    for (std::size_t t = 0; t < n_tokens; ++t) {
        const float *row = x.data() + t * hidden;
        float *dst = out.data() + t * hidden;

        // Accumulate in double: a naive fp32 sum of 2048 squares drops
        // several low bits, and every tolerance downstream inherits that.
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < hidden; ++i) {
            const double v = static_cast<double>(row[i]);
            sum_sq += v * v;
        }
        const double mean_sq = sum_sq / static_cast<double>(hidden);
        const float scale = static_cast<float>(1.0 / std::sqrt(mean_sq + eps));

        // Scale first, gain second. LlamaRMSNorm.forward() does the rsqrt
        // multiply before applying self.weight; grouping them the other way
        // rounds differently.
        for (std::size_t i = 0; i < hidden; ++i) {
            dst[i] = bf16_to_f32(g[i]) * (row[i] * scale);
        }
    }
    return out;
}

} // namespace l3llm
