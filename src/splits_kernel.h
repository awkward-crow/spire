#pragma once
#include <stdint.h>

int32_t spire_findBestBin(
    const float* restrict prefixG,
    const float* restrict prefixH,
    int32_t      nbins,
    float        G_P,
    float        H_P,
    float        lambda,
    float        minHess,
    float*       restrict out_gain,
    float*       restrict out_G_L,
    float*       restrict out_H_L);
