#include "rope.h"

#include <cmath>
#include <numbers>

#include "die.h"
#include "trace.h"

namespace l3llm {
namespace {

// Hugging Face's _compute_llama3_parameters. Bands are chosen by
// wavelength relative to the original training context: short wavelengths
// (high frequency) pass through untouched, long ones are divided by
// `factor`, and a band in between interpolates smoothly.
void apply_llama3_scaling(std::vector<double> &inv_freq, const RopeScaling &rs) {
    const double orig = static_cast<double>(rs.original_max_position_embeddings);
    const double low_wavelen = orig / rs.low_freq_factor;
    const double high_wavelen = orig / rs.high_freq_factor;

    for (double &f : inv_freq) {
        const double wavelen = 2.0 * std::numbers::pi / f;
        if (wavelen > low_wavelen) {
            f /= rs.factor;
        } else if (wavelen >= high_wavelen) {
            const double smooth =
                (orig / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor);
            f = (1.0 - smooth) * (f / rs.factor) + smooth * f;
        }
        // wavelen < high_wavelen: high frequency, left as is.
    }
}

} // namespace

RopeTable rope_table(const ModelConfig &cfg, std::int64_t n_positions) {
    const TraceScope trace("rope_table");
    if (cfg.head_dim <= 0 || cfg.head_dim % 2 != 0) {
        die("rope: head_dim {} must be positive and even", cfg.head_dim);
    }
    const std::size_t half = static_cast<std::size_t>(cfg.head_dim / 2);

    // Base frequencies: inv_freq[j] = theta^(-2j/head_dim). Low j rotates
    // fast, high j slowly, giving a spectrum of wavelengths.
    std::vector<double> inv_freq(half);
    for (std::size_t j = 0; j < half; ++j) {
        const double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(cfg.head_dim);
        inv_freq[j] = 1.0 / std::pow(cfg.rope_theta, exponent);
    }

    if (cfg.rope_scaling) {
        const std::string &type = cfg.rope_scaling->rope_type;
        if (type == "llama3") {
            apply_llama3_scaling(inv_freq, *cfg.rope_scaling);
        } else if (!type.empty() && type != "default") {
            die("rope: unsupported rope_type '{}'", type);
        }
    }

    RopeTable table;
    table.head_dim = cfg.head_dim;
    table.n_positions = n_positions;
    table.cos.resize(static_cast<std::size_t>(n_positions) * half);
    table.sin.resize(table.cos.size());
    for (std::int64_t m = 0; m < n_positions; ++m) {
        for (std::size_t j = 0; j < half; ++j) {
            const double angle = static_cast<double>(m) * inv_freq[j];
            const std::size_t k = static_cast<std::size_t>(m) * half + j;
            table.cos[k] = static_cast<float>(std::cos(angle));
            table.sin[k] = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

std::vector<float> split_heads(std::span<const float> x, std::int64_t n_heads,
                               std::int64_t head_dim) {
    const TraceScope trace("split_heads");
    const std::size_t width = static_cast<std::size_t>(n_heads * head_dim);
    if (width == 0 || x.size() % width != 0) {
        die("split_heads: {} floats is not a whole number of rows of {} heads x {}", x.size(),
            n_heads, head_dim);
    }
    const std::size_t n_tokens = x.size() / width;
    const std::size_t hd = static_cast<std::size_t>(head_dim);

    std::vector<float> out(x.size());
    for (std::size_t h = 0; h < static_cast<std::size_t>(n_heads); ++h) {
        for (std::size_t t = 0; t < n_tokens; ++t) {
            const float *src = x.data() + t * width + h * hd;
            float *dst = out.data() + (h * n_tokens + t) * hd;
            for (std::size_t i = 0; i < hd; ++i) {
                dst[i] = src[i];
            }
        }
    }
    return out;
}

void apply_rope(std::span<float> x, std::int64_t n_heads, std::int64_t n_tokens,
                const RopeTable &table) {
    const TraceScope trace("apply_rope", 3.0 * x.size());
    const std::size_t hd = static_cast<std::size_t>(table.head_dim);
    const std::size_t half = hd / 2;
    if (x.size() != static_cast<std::size_t>(n_heads * n_tokens) * hd) {
        die("apply_rope: buffer of {} floats does not match {} heads x {} tokens x {}", x.size(),
            n_heads, n_tokens, hd);
    }
    if (n_tokens > table.n_positions) {
        die("apply_rope: {} tokens exceeds table built for {} positions", n_tokens,
            table.n_positions);
    }

    for (std::size_t h = 0; h < static_cast<std::size_t>(n_heads); ++h) {
        for (std::size_t t = 0; t < static_cast<std::size_t>(n_tokens); ++t) {
            float *v = x.data() + (h * static_cast<std::size_t>(n_tokens) + t) * hd;
            const float *cos = table.cos.data() + t * half;
            const float *sin = table.sin.data() + t * half;
            // rotate_half: j pairs with j + half, not with j + 1.
            for (std::size_t j = 0; j < half; ++j) {
                const float x1 = v[j];
                const float x2 = v[j + half];
                v[j] = x1 * cos[j] - x2 * sin[j];
                v[j + half] = x2 * cos[j] + x1 * sin[j];
            }
        }
    }
}

} // namespace l3llm
