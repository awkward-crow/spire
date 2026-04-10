/*
  Objectives.chpl
  ---------------
  Gradient boosting objectives implemented as Chapel records satisfying
  the GBMObjective interface.

  Interface methods every objective must provide:
    initF(ref data: GBMData): real   — set data.F to optimal constant,
                                       return that value as baseScore
    gradients(F, y, ref g, ref h)    — compute per-sample gradient/hessian
    loss(F, y): real                 — scalar loss for training-progress logging
    defaultMinHess(): real           — minimum hessian sum for valid splits

  Supported objectives:
    MSE      — mean squared error (regression)
    LogLoss  — binary cross-entropy (classification)
    Pinball  — quantile regression; carries its own tau field

  Standalone helpers mseLoss, logLoss, pinballLoss, sigmoid, clamp remain
  public module-level procs for use in tests and drivers.
*/

module Objectives {

  use Math;
  use DataLayout;
  use Sort;

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  // Numerically stable sigmoid
  inline proc sigmoid(x: real): real {
    if x >= 0.0 then
      return 1.0 / (1.0 + exp(-x));
    else {
      const e = exp(x);
      return e / (1.0 + e);
    }
  }

  // Clamp to avoid log(0) in log-loss
  inline proc clamp(x: real, lo: real, hi: real): real {
    return min(max(x, lo), hi);
  }

  // ------------------------------------------------------------------
  // MSE
  //   Loss     : (F - y)^2
  //   Gradient : F - y
  //   Hessian  : 1  (constant)
  // ------------------------------------------------------------------
  record MSE {
    proc initF(ref data: GBMData): real {
      const baseScore = (+ reduce data.y) / data.numSamples: real;
      data.F = baseScore;
      return baseScore;
    }
    proc gradients(F: [] real, y: [] real,
                   ref g: [] real, ref h: [] real) {
      forall i in F.domain {
        g[i] = F[i] - y[i];
        h[i] = 1.0;
      }
    }
    proc loss(F: [] real, y: [] real): real { return mseLoss(F, y); }
    proc defaultMinHess(): real             { return 1.0; }
  }

  // ------------------------------------------------------------------
  // LogLoss  (binary classification, labels in {0, 1})
  //   F[i] is a raw logit; p[i] = sigmoid(F[i])
  //   Loss     : -y*log(p) - (1-y)*log(1-p)
  //   Gradient : p - y
  //   Hessian  : p * (1 - p)
  // ------------------------------------------------------------------
  record LogLoss {
    proc initF(ref data: GBMData): real {
      const pMean = (+ reduce data.y) / data.numSamples: real;
      const p     = max(1e-7, min(1.0 - 1e-7, pMean));
      const baseScore = log(p / (1.0 - p));
      data.F = baseScore;
      return baseScore;
    }
    proc gradients(F: [] real, y: [] real,
                   ref g: [] real, ref h: [] real) {
      forall i in F.domain {
        const p = sigmoid(F[i]);
        g[i] = p - y[i];
        h[i] = max(p * (1.0 - p), 1e-16);
      }
    }
    proc loss(F: [] real, y: [] real): real { return logLoss(F, y); }
    proc defaultMinHess(): real             { return 1e-6; }
  }

  // ------------------------------------------------------------------
  // Pinball (quantile regression)
  //   tau in (0, 1) is the target quantile level.
  //   Loss     : tau * max(y-F, 0)  +  (1-tau) * max(F-y, 0)
  //   Gradient : -tau      if F < y  (under-predicted)
  //              (1 - tau) if F > y  (over-predicted)
  //              0         if F == y (measure-zero)
  //   Hessian  : constant 1.0  (standard GBM approximation — true hessian
  //              of piecewise-linear loss is 0 a.e., but GBM needs positive
  //              curvature; 1.0 matches LightGBM and sklearn)
  // ------------------------------------------------------------------
  record Pinball {
    var tau: real;

    proc initF(ref data: GBMData): real {
      var yLocal: [0..#data.numSamples] real;
      forall i in data.rowDom do yLocal[i] = data.y[i];
      sort(yLocal);
      const qIdx = min((tau * data.numSamples: real): int,
                       data.numSamples - 1);
      const baseScore = yLocal[qIdx];
      data.F = baseScore;
      return baseScore;
    }

    proc gradients(F: [] real, y: [] real,
                   ref g: [] real, ref h: [] real) {
      if tau <= 0.0 || tau >= 1.0 then
        halt("Pinball tau must be in (0, 1), got: " + tau:string);
      forall i in F.domain {
        const residual = y[i] - F[i];
        if residual > 0.0 then
          g[i] = -tau;
        else if residual < 0.0 then
          g[i] = 1.0 - tau;
        else
          g[i] = 0.0;
        h[i] = 1.0;
      }
    }

    proc loss(F: [] real, y: [] real): real { return pinballLoss(F, y, tau); }
    proc defaultMinHess(): real             { return 1.0; }
  }

  // ------------------------------------------------------------------
  // Standalone loss functions — useful in tests and drivers
  // ------------------------------------------------------------------

  proc mseLoss(F: [] real, y: [] real): real {
    return (+ reduce [(i) in F.domain] (F[i] - y[i])**2) / F.size: real;
  }

  proc logLoss(F: [] real, y: [] real): real {
    const eps = 1e-15;
    var s: real = 0.0;
    forall i in F.domain with (+ reduce s) {
      const p = clamp(sigmoid(F[i]), eps, 1.0 - eps);
      s += -(y[i] * log(p) + (1.0 - y[i]) * log(1.0 - p));
    }
    return s / F.size: real;
  }

  proc pinballLoss(F: [] real, y: [] real, tau: real): real {
    var s: real = 0.0;
    forall i in F.domain with (+ reduce s) {
      const r = y[i] - F[i];
      s += if r > 0.0 then tau * r else (tau - 1.0) * r;
    }
    return s / F.size: real;
  }

} // module Objectives
