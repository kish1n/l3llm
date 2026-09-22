#include "mlp.h"

#include <cmath>
#include <cstddef>

#include "die.h"
#include "linear.h"

namespace qllm {
namespace {

// z * sigmoid(z). Safe at both extremes: a large positive z gives
// exp(-z) -> 0, a large negative z gives exp(-z) -> inf and the quotient
// underflows to 0 rather than overflowing.
inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

} // namespace

std::vector<float> mlp(std::span<const float> h, const TensorView &gate, const TensorView &up,
                       const TensorView &down) {
    std::vector<float> g = linear(h, gate);
    const std::vector<float> u = linear(h, up);
    if (g.size() != u.size()) {
        die("mlp: gate produced {} floats but up produced {}", g.size(), u.size());
    }

    // Fuse the activation and the gate into one pass over the 8192-wide
    // intermediate, reusing g's buffer rather than allocating a third.
    for (std::size_t i = 0; i < g.size(); ++i) {
        g[i] = silu(g[i]) * u[i];
    }
    return linear(g, down);
}

} // namespace qllm
