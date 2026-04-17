/*
  splits_kernel.c
  ---------------
  Vectorizable best-split gain kernel for the 255-bin inner loop.

  Kept in a separate translation unit so LLVM cannot inline it into
  the Chapel runtime wrapper, which previously prevented auto-vectorization
  even though the pre-inlined IR showed a perfectly clean vectorizable loop.

  Two passes (caller supplies pass 1 prefix arrays):
    Pass 1 (Chapel): sequential prefix scan, loop-carried dep on gAcc/hAcc,
                     writes contiguous prefixG[]/prefixH[] for pass 2.
    Pass 2 (here):   unconditional gain compute → gains[] array (vectorizable),
                     then scalar argmax (255 iters, no divides).

  With -O3 -march=native LLVM emits vdivps (16-wide AVX-512) or vdivps ymm
  (8-wide AVX2) for the gain loop, replacing ~510 scalar fdivss per feature.
*/

#include "splits_kernel.h"

__attribute__((noinline))
int32_t spire_findBestBin(
    const float* restrict prefixG,   /* [nbins] prefix grad sums     */
    const float* restrict prefixH,   /* [nbins] prefix hess sums     */
    int32_t      nbins,
    float        G_P,                /* parent grad total            */
    float        H_P,                /* parent hess total            */
    float        lambda,             /* L2 regularisation            */
    float        minHess,            /* min hess per child           */
    float*       restrict out_gain,
    float*       restrict out_G_L,
    float*       restrict out_H_L
) {
    float score_P = G_P * G_P / (H_P + lambda);

    /*
     * Vectorizable loop: no loop-carried arithmetic deps, stride-1 reads,
     * unconditional compute, branchless validity mask.
     * gains[256] aligned to 64 bytes → fits three AVX-512 registers.
     */
    float gains[256] __attribute__((aligned(64)));
    for (int32_t b = 0; b < nbins; b++) {
        float G_L  = prefixG[b];
        float H_L  = prefixH[b];
        float G_R  = G_P - G_L;
        float H_R  = H_P - H_L;
        float gain = G_L * G_L / (H_L + lambda)
                   + G_R * G_R / (H_R + lambda)
                   - score_P;
        /* branchless AND keeps both comparisons as predicate ops, no branch */
        int valid  = (H_L >= minHess) & (H_R >= minHess);
        gains[b]   = valid ? gain : -1.0f;
    }

    /* Scalar argmax — no divides, 255 compare+select iterations. */
    int32_t best_b    = -1;
    float   best_gain = 0.0f;
    float   best_G_L  = 0.0f;
    float   best_H_L  = 0.0f;
    for (int32_t b = 0; b < nbins; b++) {
        if (gains[b] > best_gain) {
            best_gain = gains[b];
            best_b    = b;
            best_G_L  = prefixG[b];
            best_H_L  = prefixH[b];
        }
    }

    *out_gain = best_gain;
    *out_G_L  = best_G_L;
    *out_H_L  = best_H_L;
    return best_b;
}
