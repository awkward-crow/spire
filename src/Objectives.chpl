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
  use Tree;

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  // Numerically stable sigmoid — float64 and float32 overloads
  inline proc sigmoid(x: real): real {
    if x >= 0.0 then
      return 1.0 / (1.0 + exp(-x));
    else {
      const e = exp(x);
      return e / (1.0 + e);
    }
  }

  inline proc sigmoid(x: real(32)): real(32) {
    if x >= 0.0: real(32) then
      return 1.0: real(32) / (1.0: real(32) + exp(-x));
    else {
      const e = exp(x);
      return e / (1.0: real(32) + e);
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
      const baseScore = (+ reduce data.y): real / data.numSamples: real;
      data.F = baseScore;
      return baseScore;
    }
    proc gradients(F: [] real, y: [] real(32),
                   ref g: [] real(32), ref h: [] real(32)) {
      forall i in F.domain {
        g[i] = F[i]: real(32) - y[i];
        h[i] = 1.0: real(32);
      }
    }
    proc loss(F: [] real, y: [] real(32)): real { return mseLoss(F, y); }
    proc defaultMinHess(): real                 { return 1.0; }
    // Newton step is the exact minimiser for MSE — no refit needed.
    proc leafRefit(ref tree: FittedTree, nodeId: [] int,
                   F: [] real, y: [] real(32), eta: real) { }
  }

  // ------------------------------------------------------------------
  // LogLoss  (binary classification, labels in {0, 1})
  //   F[i] is a raw logit; p[i] = sigmoid(F[i])
  //   Loss     : -y*log(p) - (1-y)*log(1-p)
  //   Gradient : p - y
  //   Hessian  : p * (1 - p)
  // ------------------------------------------------------------------
  record LogLoss {
    // Minimum hessian sum for a valid leaf.  p*(1-p) can be tiny when the
    // model is confident, so the default is much smaller than MSE's 1.0.
    // Override at construction time: new LogLoss(minHess=0.25).
    var minHess: real = 1e-6;

    proc initF(ref data: GBMData): real {
      const pMean = (+ reduce data.y): real / data.numSamples: real;
      const p     = max(1e-7, min(1.0 - 1e-7, pMean));
      const baseScore = log(p / (1.0 - p));
      data.F = baseScore;
      return baseScore;
    }
    proc gradients(F: [] real, y: [] real(32),
                   ref g: [] real(32), ref h: [] real(32)) {
      forall i in F.domain {
        const p = sigmoid(F[i]: real(32));
        g[i] = p - y[i];
        h[i] = max(p * (1.0: real(32) - p), 1e-16: real(32));
      }
    }
    proc loss(F: [] real, y: [] real(32)): real { return logLoss(F, y); }
    proc defaultMinHess(): real                 { return minHess; }
    // Newton step is well-defined for LogLoss — no refit needed.
    proc leafRefit(ref tree: FittedTree, nodeId: [] int,
                   F: [] real, y: [] real(32), eta: real) { }
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
      forall i in data.rowDom do yLocal[i] = data.y[i]: real;
      sort(yLocal);
      const qIdx = min((tau * data.numSamples: real): int,
                       data.numSamples - 1);
      const baseScore = yLocal[qIdx];
      data.F = baseScore;
      return baseScore;
    }

    proc gradients(F: [] real, y: [] real(32),
                   ref g: [] real(32), ref h: [] real(32)) {
      if tau <= 0.0 || tau >= 1.0 then
        halt("Pinball tau must be in (0, 1), got: " + tau:string);
      const hessVal: real(32) = (tau * (1.0 - tau)): real(32);
      forall i in F.domain {
        const residual = y[i]: real - F[i];   // float64 comparison
        if residual > 0.0 then
          g[i] = (-tau): real(32);
        else if residual < 0.0 then
          g[i] = (1.0 - tau): real(32);
        else
          g[i] = 0.0: real(32);
        h[i] = hessVal;
      }
    }

    proc loss(F: [] real, y: [] real(32)): real { return pinballLoss(F, y, tau); }
    // minHess = tau*(1-tau) keeps the 1-sample-per-leaf threshold consistent
    // with the new hessian scale.  Returning 1.0 would require ~1/(tau*(1-tau))
    // samples minimum, which over-prunes for extreme tau values.
    proc defaultMinHess(): real             { return tau * (1.0 - tau); }

    // Post-hoc leaf refit: replace the Newton-step leaf value with the
    // tau-quantile of residuals (y_i - F_i) in the leaf.
    //
    // This is LightGBM's RenewTreeOutput() approach.  The Newton step is
    // bounded by O(1/(1-tau)) regardless of residual magnitude; the quantile
    // of actual residuals can be orders of magnitude larger, enabling
    // convergence in far fewer trees.
    //
    // nodeId[i] contains the leaf heap index for sample i — valid after
    // finalizeLeaves, before applyTree is called.
    proc leafRefit(ref tree: FittedTree, nodeId: [] int,
                   F: [] real, y: [] real(32), eta: real) {
      if tau <= 0.0 || tau >= 1.0 then
        halt("Pinball tau must be in (0, 1), got: " + tau:string);

      const nSamples = nodeId.domain.size;

      // Gather distributed nodeId and residuals to locale 0.
      // Follows the same pattern as Pinball.initF.
      var localNodeId  : [0..#nSamples] int;
      var localResidual: [0..#nSamples] real;
      forall i in nodeId.domain {
        localNodeId[i]   = nodeId[i];
        localResidual[i] = y[i]: real - F[i];
      }

      // Scratch buffer — reused across leaves; sized to worst case (all
      // samples in one leaf).
      var leafBuf: [0..#nSamples] real;

      for nodeIdx in tree.nodeDom {
        if !tree.isLeaf[nodeIdx] then continue;

        // Collect residuals for this leaf in a single pass.
        var cnt = 0;
        for i in 0..#nSamples {
          if localNodeId[i] == nodeIdx {
            leafBuf[cnt] = localResidual[i];
            cnt += 1;
          }
        }
        if cnt == 0 then continue;   // phantom leaf — keep value = 0

        // Sort the slice and take the tau-quantile.
        var residuals: [0..#cnt] real = leafBuf[0..#cnt];
        sort(residuals);
        const qIdx = min((tau * cnt: real): int, cnt - 1);
        tree.value[nodeIdx] = eta * residuals[qIdx];
      }
    }
  }

  // ------------------------------------------------------------------
  // Standalone loss functions — useful in tests and drivers
  // ------------------------------------------------------------------

  proc mseLoss(F: [] real, y: [] real(32)): real {
    return (+ reduce [(i) in F.domain] (F[i] - y[i]: real)**2) / F.size: real;
  }

  proc logLoss(F: [] real, y: [] real(32)): real {
    const eps = 1e-15;
    var s: real = 0.0;
    forall i in F.domain with (+ reduce s) {
      const p = clamp(sigmoid(F[i]), eps, 1.0 - eps);
      s += -(y[i]: real * log(p) + (1.0 - y[i]: real) * log(1.0 - p));
    }
    return s / F.size: real;
  }

  proc pinballLoss(F: [] real, y: [] real(32), tau: real): real {
    var s: real = 0.0;
    forall i in F.domain with (+ reduce s) {
      const r = y[i]: real - F[i];
      s += if r > 0.0 then tau * r else (tau - 1.0) * r;
    }
    return s / F.size: real;
  }

} // module Objectives
