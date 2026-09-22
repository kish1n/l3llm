#include "model.h"

#include <cstddef>

#include "die.h"
#include "embed.h"
#include "layer.h"
#include "linear.h"
#include "rmsnorm.h"
#include "rope.h"

namespace qllm {

ForwardResult forward(const SafeTensors &weights, const ModelConfig &cfg,
                      std::span<const std::int32_t> token_ids, const LayerHook &on_layer) {
    if (token_ids.empty()) {
        die("forward: no tokens");
    }
    const std::int64_t n_tokens = static_cast<std::int64_t>(token_ids.size());

    std::vector<float> x = embed(weights, cfg, token_ids);

    // Angles depend only on the config, so one table serves every layer.
    const RopeTable table = rope_table(cfg, n_tokens);
    for (std::int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        decoder_layer(x, weights, cfg, i, table, n_tokens);
        if (on_layer) {
            on_layer(i, x);
        }
    }

    ForwardResult out;
    out.hidden = rmsnorm(x, weights.at("model.norm.weight"), cfg.rms_norm_eps);

    const TensorView &head = cfg.tie_word_embeddings ? weights.at("model.embed_tokens.weight")
                                                     : weights.at("lm_head.weight");
    out.logits = linear(out.hidden, head);
    return out;
}

std::int32_t argmax_last(const ForwardResult &out, const ModelConfig &cfg) {
    const std::size_t vocab = static_cast<std::size_t>(cfg.vocab_size);
    if (vocab == 0 || out.logits.size() % vocab != 0) {
        die("argmax_last: {} logits is not a whole number of rows of {}", out.logits.size(), vocab);
    }
    const float *row = out.logits.data() + (out.logits.size() - vocab);

    std::size_t best = 0;
    for (std::size_t i = 1; i < vocab; ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return static_cast<std::int32_t>(best);
}

} // namespace qllm
