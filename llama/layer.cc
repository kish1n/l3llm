#include "layer.h"

#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include "attention.h"
#include "die.h"
#include "linear.h"
#include "mlp.h"
#include "rmsnorm.h"
#include "trace.h"

namespace qllm {
namespace {

void add_into(std::span<float> x, std::span<const float> delta) {
    const TraceScope trace("residual_add", static_cast<double>(x.size()));
    if (x.size() != delta.size()) {
        die("residual: cannot add {} floats into {}", delta.size(), x.size());
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] += delta[i];
    }
}

} // namespace

void decoder_layer(std::span<float> x, const SafeTensors &weights, const ModelConfig &cfg,
                   std::int64_t index, const RopeTable &table, std::int64_t n_tokens) {
    const TraceScope trace(TraceSession::enabled() ? std::format("layer {}", index) : "layer");
    const auto w = [&](std::string_view suffix) -> const TensorView & {
        return weights.at(std::format("model.layers.{}.{}", index, suffix));
    };

    // --- attention branch ---
    {
        const TraceScope attention_trace("self_attention");
        const std::vector<float> h = rmsnorm(x, w("input_layernorm.weight"), cfg.rms_norm_eps);

        std::vector<float> q = split_heads(linear(h, w("self_attn.q_proj.weight")),
                                           cfg.num_attention_heads, cfg.head_dim);
        std::vector<float> k = split_heads(linear(h, w("self_attn.k_proj.weight")),
                                           cfg.num_key_value_heads, cfg.head_dim);
        // V is split into heads but never rotated.
        const std::vector<float> v = split_heads(linear(h, w("self_attn.v_proj.weight")),
                                                 cfg.num_key_value_heads, cfg.head_dim);

        apply_rope(q, cfg.num_attention_heads, n_tokens, table);
        apply_rope(k, cfg.num_key_value_heads, n_tokens, table);

        const std::vector<float> ctx = attention(q, k, v, cfg, n_tokens);
        const std::vector<float> merged =
            merge_heads(ctx, cfg.num_attention_heads, n_tokens, cfg.head_dim);
        add_into(x, linear(merged, w("self_attn.o_proj.weight")));
    }

    // --- MLP branch, reading x as updated above ---
    {
        const TraceScope mlp_trace("feed_forward");
        const std::vector<float> h =
            rmsnorm(x, w("post_attention_layernorm.weight"), cfg.rms_norm_eps);
        add_into(x, mlp(h, w("mlp.gate_proj.weight"), w("mlp.up_proj.weight"),
                        w("mlp.down_proj.weight")));
    }
}

} // namespace qllm
