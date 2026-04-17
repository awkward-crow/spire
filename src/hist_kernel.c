/*
  hist_kernel.c
  -------------
  4-sample unrolled histogram scatter kernel for GBM tree building.

  The histogram inner loop (one sample pass, one feature at a time) is:

      for each sample i:
          if slots[i] >= 0:
              lghOut[bins[i]][slots[i]].grad += grad[i]
              lghOut[bins[i]][slots[i]].hess += hess[i]

  This is a random scatter: the destination address depends on bins[i],
  which is data-dependent.  A scalar loop can sustain at most ~1 scatter
  per 4–12 cycles due to load-to-use latency.

  By reading 4 samples per iteration, we issue 4 *independent* scatter
  ops simultaneously.  The CPU can overlap their address computations and
  cache-line fetches, improving throughput to ~1 effective scatter per
  1–3 cycles (2–3× improvement in the scatter-dominated case).

  lghOut layout: flat float[2 * MAX_BINS * nSlots], interleaved GH pairs:
    lghOut[2*(b*nSlots + s) + 0] = grad accumulator for (bin b, slot s)
    lghOut[2*(b*nSlots + s) + 1] = hess accumulator for (bin b, slot s)

  This matches Chapel's row-major layout for a [MAX_BINS, nSlots] GH array
  since GH = {float grad; float hess;} — 8 bytes, 2 consecutive floats.

  Kept in a separate translation unit so LLVM cannot inline it into Chapel's
  coforall wrapper (same reason as splits_kernel.c).
*/

#include "hist_kernel.h"

__attribute__((noinline))
void spire_histKernel(
    const float*   restrict grad,
    const float*   restrict hess,
    const uint8_t* restrict bins,
    const int32_t* restrict slots,
    int32_t        nSamples,
    int32_t        nSlots,
    float*         restrict lghOut
) {
    int32_t i = 0;

    /* 4-unrolled main loop.
     * All four loads (s0..s3, b0..b3) and pointer computations are
     * independent — the CPU can issue them in parallel, hiding the
     * load latency for the scatter targets. */
    for (; i + 3 < nSamples; i += 4) {
        int32_t s0 = slots[i],     s1 = slots[i+1];
        int32_t s2 = slots[i+2],   s3 = slots[i+3];
        int32_t b0 = (int32_t)bins[i],   b1 = (int32_t)bins[i+1];
        int32_t b2 = (int32_t)bins[i+2], b3 = (int32_t)bins[i+3];

        if (s0 >= 0) {
            float* p = lghOut + 2 * (b0 * nSlots + s0);
            p[0] += grad[i];
            p[1] += hess[i];
        }
        if (s1 >= 0) {
            float* p = lghOut + 2 * (b1 * nSlots + s1);
            p[0] += grad[i+1];
            p[1] += hess[i+1];
        }
        if (s2 >= 0) {
            float* p = lghOut + 2 * (b2 * nSlots + s2);
            p[0] += grad[i+2];
            p[1] += hess[i+2];
        }
        if (s3 >= 0) {
            float* p = lghOut + 2 * (b3 * nSlots + s3);
            p[0] += grad[i+3];
            p[1] += hess[i+3];
        }
    }

    /* Scalar tail for remaining 0–3 samples. */
    for (; i < nSamples; i++) {
        int32_t s = slots[i];
        if (s >= 0) {
            float* p = lghOut + 2 * ((int32_t)bins[i] * nSlots + s);
            p[0] += grad[i];
            p[1] += hess[i];
        }
    }
}
