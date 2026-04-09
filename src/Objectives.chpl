/*
  Objectives.chpl
  ---------------
  Gradient and hessian computation for GBM training objectives.

  Supported:
    - MSE          : mean squared error (regression baseline)
    - LogLoss      : binary cross-entropy (classification)
    - Pinball      : quantile regression (asymmetric pinball loss)

  All procedures operate on distributed arrays so they work identically
  single-locale and multi-locale — the forall loops will simply run
  locally when numLocales == 1.
*/

module Objectives {

  use Math;

  // ------------------------------------------------------------------
  // Objective tag — passed through to the boosting loop so it knows
  // which gradient function to call each round.
  // ------------------------------------------------------------------
  enum Objective { MSE, LogLoss, Pinball }

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

  // Clamp predictions to avoid log(0) in log-loss
  inline proc clamp(x: real, lo: real, hi: real): real {
    return min(max(x, lo), hi);
  }

  // ------------------------------------------------------------------
  // MSE
  //   Loss     : 0.5 * (F - y)^2
  //   Gradient : F - y
  //   Hessian  : 1  (constant — standard in GBM)
  // ------------------------------------------------------------------
  proc computeGradients_MSE(
      F    : [] real,       // current predictions
      y    : [] real,       // targets
      ref grad : [] real,   // output: gradient per sample
      ref hess : [] real    // output: hessian per sample
  ) {
    forall i in F.domain {
      grad[i] = F[i] - y[i];
      hess[i] = 1.0;
    }
  }

  // ------------------------------------------------------------------
  // Log-loss  (binary classification, labels in {0, 1})
  //   F[i] is a raw logit (unbounded real)
  //   p[i] = sigmoid(F[i])
  //   Loss     : -y*log(p) - (1-y)*log(1-p)
  //   Gradient : p - y
  //   Hessian  : p * (1 - p)
  // ------------------------------------------------------------------
  proc computeGradients_LogLoss(
      F        : [] real,
      y        : [] real,
      ref grad : [] real,
      ref hess : [] real
  ) {
    forall i in F.domain {
      const p = sigmoid(F[i]);
      grad[i] = p - y[i];
      hess[i] = max(p * (1.0 - p), 1e-16);  // floor avoids zero hessian
    }
  }

  // ------------------------------------------------------------------
  // Pinball (quantile regression)
  //   tau in (0, 1) is the target quantile, e.g. 0.9 for 90th percentile
  //
  //   Loss     : tau * max(y-F, 0)  +  (1-tau) * max(F-y, 0)
  //   Gradient : -(tau)      if F < y   (under-predicted)
  //              +(1 - tau)  if F > y   (over-predicted)
  //              0           if F == y  (measure-zero, treat as 0)
  //   Hessian  : constant 1.0  (pinball is piecewise linear — true
  //              hessian is 0 a.e., but GBM needs a positive curvature
  //              estimate; 1.0 is the standard approximation used by
  //              LightGBM and sklearn's HistGradientBoosting)
  //
  //   Note on sign convention: we follow XGBoost/LightGBM — gradient
  //   is the derivative of the *loss* w.r.t. F, so a negative gradient
  //   means the tree should push F *up*.
  // ------------------------------------------------------------------
  proc computeGradients_Pinball(
      F        : [] real,
      y        : [] real,
      ref grad : [] real,
      ref hess : [] real,
      tau      : real        // quantile level, must be in (0, 1)
  ) {
    if tau <= 0.0 || tau >= 1.0 then
      halt("Pinball tau must be in (0, 1), got: " + tau:string);

    forall i in F.domain {
      const residual = y[i] - F[i];
      if residual > 0.0 then
        grad[i] = -tau;           // under-predicted: loss gradient is -tau
      else if residual < 0.0 then
        grad[i] = 1.0 - tau;     // over-predicted: loss gradient is (1-tau)
      else
        grad[i] = 0.0;
      hess[i] = 1.0;
    }
  }

  // ------------------------------------------------------------------
  // Loss functions — scalar evaluation over prediction/target arrays.
  // Useful for monitoring training progress and evaluating on test sets.
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

  // Unified loss dispatch — mirrors computeGradients
  proc computeLoss(
      obj : Objective,
      F   : [] real,
      y   : [] real,
      tau : real = 0.5
  ): real {
    select obj {
      when Objective.MSE     do return mseLoss(F, y);
      when Objective.LogLoss do return logLoss(F, y);
      when Objective.Pinball do return pinballLoss(F, y, tau);
      otherwise halt("Unknown objective");
    }
    return 0.0;
  }

  // ------------------------------------------------------------------
  // Unified dispatch — call this from the boosting loop
  // ------------------------------------------------------------------
  proc computeGradients(
      obj      : Objective,
      F        : [] real,
      y        : [] real,
      ref grad : [] real,
      ref hess : [] real,
      tau      : real = 0.5   // only used when obj == Pinball
  ) {
    select obj {
      when Objective.MSE     do computeGradients_MSE(F, y, grad, hess);
      when Objective.LogLoss do computeGradients_LogLoss(F, y, grad, hess);
      when Objective.Pinball do computeGradients_Pinball(F, y, grad, hess, tau);
      otherwise halt("Unknown objective");
    }
  }

} // module Objectives
