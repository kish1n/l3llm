#include "attention.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include "die.h"
#include "trace.h"

namespace qllm {

std::vector<float> attention(std::span<const float> q, std::span<const float> k,
                             std::span<const float> v, const ModelConfig &cfg,
                             std::int64_t n_tokens) {
    const std::size_t hd = static_cast<std::size_t>(cfg.head_dim);
    const std::size_t nh = static_cast<std::size_t>(cfg.num_attention_heads);
    const std::size_t nkv = static_cast<std::size_t>(cfg.num_key_value_heads);
    const std::size_t nt = static_cast<std::size_t>(n_tokens);
    const std::size_t group = static_cast<std::size_t>(cfg.heads_per_kv_group());
    const bool detailed = TraceSession::detailed_attention();
    const double pairs = static_cast<double>(nt) * (static_cast<double>(nt) + 1) / 2;
    // Detailed children contribute their counts to this parent. In coarse
    // mode count the same work analytically, including the scale division.
    const TraceScope trace("attention", detailed ? 1.0 : 1.0 + nh * (pairs * (4.0 * hd + 4) + nt));

    if (q.size() != nh * nt * hd) {
        die("attention: q has {} floats, expected {} heads x {} tokens x {}", q.size(), nh, nt, hd);
    }
    if (k.size() != nkv * nt * hd || v.size() != nkv * nt * hd) {
        die("attention: k/v have {}/{} floats, expected {} kv heads x {} tokens x {}", k.size(),
            v.size(), nkv, nt, hd);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<float> out(nh * nt * hd, 0.0f);
    std::vector<float> scores(nt);

    for (std::size_t h = 0; h < nh; ++h) {
        const std::size_t kv = h / group;
        const float *qh = q.data() + h * nt * hd;
        const float *kh = k.data() + kv * nt * hd;
        const float *vh = v.data() + kv * nt * hd;

        for (std::size_t i = 0; i < nt; ++i) {
            const float *qi = qh + i * hd;

            // Causal: j runs to i inclusive, so future keys are never
            // scored at all.
            float row_max = -std::numeric_limits<float>::infinity();
            {
                const TraceScope scores_trace("attention_scores", (2.0 * hd + 1) * (i + 1), {},
                                              detailed);
                for (std::size_t j = 0; j <= i; ++j) {
                    const float *kj = kh + j * hd;
                    float acc = 0.0f;
                    for (std::size_t d = 0; d < hd; ++d) {
                        acc += qi[d] * kj[d];
                    }
                    scores[j] = acc * scale;
                    row_max = std::max(row_max, scores[j]);
                }
            }

            // Subtract the row max before exp. Mathematically a no-op --
            // it cancels in the ratio -- but without it a large score
            // overflows to inf and the row becomes NaN.
            float inv_sum;
            {
                // Normalization is fused into weighted_values below.
                const TraceScope softmax_trace("softmax", 2.0 * (i + 1) + 1, {}, detailed);
                double sum = 0.0;
                for (std::size_t j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - row_max);
                    sum += static_cast<double>(scores[j]);
                }
                inv_sum = static_cast<float>(1.0 / sum);
            }

            const TraceScope values_trace("weighted_values", (2.0 * hd + 1) * (i + 1), {},
                                          detailed);
            float *oi = out.data() + (h * nt + i) * hd;
            for (std::size_t j = 0; j <= i; ++j) {
                const float w = scores[j] * inv_sum;
                const float *vj = vh + j * hd;
                for (std::size_t d = 0; d < hd; ++d) {
                    oi[d] += w * vj[d];
                }
            }
        }
    }
    return out;
}

std::vector<float> merge_heads(std::span<const float> x, std::int64_t n_heads,
                               std::int64_t n_tokens, std::int64_t head_dim) {
    const TraceScope trace("merge_heads");
    const std::size_t nh = static_cast<std::size_t>(n_heads);
    const std::size_t nt = static_cast<std::size_t>(n_tokens);
    const std::size_t hd = static_cast<std::size_t>(head_dim);
    if (x.size() != nh * nt * hd) {
        die("merge_heads: {} floats does not match {} heads x {} tokens x {}", x.size(), nh, nt,
            hd);
    }

    std::vector<float> out(x.size());
    for (std::size_t h = 0; h < nh; ++h) {
        for (std::size_t t = 0; t < nt; ++t) {
            const float *src = x.data() + (h * nt + t) * hd;
            float *dst = out.data() + t * nh * hd + h * hd;
            for (std::size_t d = 0; d < hd; ++d) {
                dst[d] = src[d];
            }
        }
    }
    return out;
}

} // namespace qllm
