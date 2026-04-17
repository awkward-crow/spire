#pragma once
#include <stdint.h>

/* spire_histKernel — 4-sample unrolled histogram accumulator.
 *
 * Processes nSamples samples for one feature, scattering (grad, hess)
 * pairs into lghOut indexed by (bin, slot).
 *
 * lghOut is a flat float array representing MAX_BINS * nSlots GH pairs
 * laid out as: lghOut[2*(b*nSlots + s) + 0] = grad accumulator
 *              lghOut[2*(b*nSlots + s) + 1] = hess accumulator
 *
 * Must be pre-zeroed by the caller (Chapel default-initialises GH to 0.0).
 *
 * slots: per-sample slot index in 0..nSlots-1.  -1 = inactive sample.
 * bins:  per-sample bin index (uint8, 0..MAX_BINS-1).
 */
void spire_histKernel(
    const float*   restrict grad,
    const float*   restrict hess,
    const uint8_t* restrict bins,
    const int32_t* restrict slots,
    int32_t        nSamples,
    int32_t        nSlots,
    float*         restrict lghOut
);
