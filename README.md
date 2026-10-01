# l3llm
cpu sram-only inference engine

## Operator timeline

Build with optimization and debug symbols, then record one forward pass:

```sh
cmake -S . -B build/profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/profile -j
build/profile/llama/llama --trace build/forward.json --warmup 1 \
  /path/to/model 128000 791 4062
```

Open [ui.perfetto.dev](https://ui.perfetto.dev), choose **Open trace file**, and
load `build/forward.json`. Expand the `l3llm` / `forward thread` track. The nested
bars show the forward pass, numbered layers, attention, MLP, projections,
normalization, RoPE, residual adds, and output head. Hover a compute bar to see
its estimated GFLOP/s in the label; select it for the full tensor name, estimated
FLOPs, duration, and unrounded rate in **Arguments**. Memory-only operations and
uncounted setup show `FLOP/s n/a`.

Add `--trace-attention` for per-head/per-token `attention_scores`, `softmax`, and
`weighted_values` bars. This emits three extra events per query head per token
per layer, and can noticeably perturb short operations. Softmax normalization
is fused into the weighted-values loop; the `softmax` bar covers exponentials,
the sum, and its reciprocal. Start with a short prompt for detailed traces.

Rates are **modeled arithmetic FLOPs / inclusive wall-clock duration**, not
hardware-counter measurements. Dense projections count `2 * tokens * in * out`;
multiply-add counts as two operations. Other instrumented kernels count scalar
adds, subtracts, multiplies and divides, excluding comparisons, conversions,
unary sign changes, transcendental functions (`exp`, `sqrt`, etc.), and RoPE
table setup. In particular, softmax's rate excludes the work inside `exp`.
Parent regions sum child counts without double counting and include allocation,
tracing overhead, and other work in their elapsed time. Parent and child rates
must not be added together.

Tracing is off by default, with no trace clocks or event allocation when off.
The recorder currently follows the calling thread, matching the single-threaded
kernels. JSON serialization happens after the forward pass. Model loading and
top-token printing are outside the trace. `--warmup N` runs N untraced forwards
first; the default is zero, so a first-pass trace can include weight page faults.
Warmup does not guarantee that weights fit in cache.

Run checks with:

```sh
L3LLM_MODEL_DIR=/path/to/model ctest --test-dir build/profile --output-on-failure
```

The trace test uses small synthetic inputs and runs without model weights; the
golden model tests require `L3LLM_MODEL_DIR` and `reference/golden`.
