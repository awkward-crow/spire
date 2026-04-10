/*
  TestObjectives.chpl
  -------------------
  Validates gradient and hessian computations for all three objectives
  against known analytical values on small hand-crafted inputs.

  Run single-locale:
    chpl TestObjectives.chpl -M ../src -o test_obj
    ./test_obj

  Run multi-locale (e.g. 4 locales with gasnet):
    chpl TestObjectives.chpl -M ../src -o test_obj
    ./test_obj -nl 4

  Expected output: all assertions pass, summary printed per objective.
*/

use Objectives;
use DataLayout;
use SyntheticData;
use Math;

// ------------------------------------------------------------------
// Small tolerance for floating-point comparisons
// ------------------------------------------------------------------
const EPS = 1e-10;

proc assertClose(name: string, got: real, expected: real, tol: real = EPS) {
  const err = abs(got - expected);
  if err > tol {
    writeln("FAIL  ", name, " | got=", got, " expected=", expected, " err=", err);
  } else {
    writeln("PASS  ", name);
  }
}

// ------------------------------------------------------------------
// Test MSE gradients
// ------------------------------------------------------------------
proc testMSE() {
  writeln("\n--- MSE ---");

  const n = 4;
  var F    : [0..#n] real = [ 1.0,  0.5, -0.5,  2.0];
  var y    : [0..#n] real = [ 1.0,  1.0,  0.0,  0.5];
  var grad : [0..#n] real;
  var hess : [0..#n] real;

  var mse = new MSE();
  mse.gradients(F, y, grad, hess);

  // grad[i] = F[i] - y[i]
  assertClose("MSE grad[0]",  grad[0],  0.0);   //  1.0 - 1.0
  assertClose("MSE grad[1]",  grad[1], -0.5);   //  0.5 - 1.0
  assertClose("MSE grad[2]",  grad[2], -0.5);   // -0.5 - 0.0
  assertClose("MSE grad[3]",  grad[3],  1.5);   //  2.0 - 0.5

  // hessians are all 1.0
  for i in 0..#n do
    assertClose("MSE hess[" + i:string + "]", hess[i], 1.0);
}

// ------------------------------------------------------------------
// Test LogLoss gradients
// ------------------------------------------------------------------
proc testLogLoss() {
  writeln("\n--- LogLoss ---");

  // sigmoid(0) = 0.5, sigmoid(large positive) ≈ 1, sigmoid(large neg) ≈ 0
  const n = 3;
  var F    : [0..#n] real = [ 0.0,   10.0,  -10.0 ];
  var y    : [0..#n] real = [ 1.0,    1.0,    0.0  ];
  var grad : [0..#n] real;
  var hess : [0..#n] real;

  var ll = new LogLoss();
  ll.gradients(F, y, grad, hess);

  // grad[0]: sigmoid(0) - 1 = 0.5 - 1.0 = -0.5
  assertClose("LogLoss grad[0]", grad[0], -0.5);

  // grad[1]: sigmoid(10) ≈ 1.0, grad ≈ 0
  assertClose("LogLoss grad[1]", grad[1], sigmoid(10.0) - 1.0, 1e-6);

  // grad[2]: sigmoid(-10) ≈ 0.0, y=0, grad ≈ 0
  assertClose("LogLoss grad[2]", grad[2], sigmoid(-10.0) - 0.0, 1e-6);

  // hess[0]: 0.5 * 0.5 = 0.25
  assertClose("LogLoss hess[0]", hess[0], 0.25);

  // hess should be positive everywhere
  for i in 0..#n do
    if hess[i] <= 0.0 then
      writeln("FAIL  LogLoss hess[", i, "] not positive: ", hess[i]);
}

// ------------------------------------------------------------------
// Test Pinball gradients
// ------------------------------------------------------------------
proc testPinball() {
  writeln("\n--- Pinball ---");

  // tau = 0.9 (90th percentile)
  // Under-prediction (F < y): grad = -tau = -0.9
  // Over-prediction  (F > y): grad = 1 - tau = 0.1
  // Exact (F == y):           grad = 0.0
  const tau = 0.9;
  const n = 3;
  var F    : [0..#n] real = [ 0.5,  1.5,  1.0 ];
  var y    : [0..#n] real = [ 1.0,  1.0,  1.0 ];
  var grad : [0..#n] real;
  var hess : [0..#n] real;

  var pb = new Pinball(tau=tau);
  pb.gradients(F, y, grad, hess);

  assertClose("Pinball grad[0] (under)", grad[0], -0.9);   // F=0.5 < y=1.0
  assertClose("Pinball grad[1] (over)",  grad[1],  0.1);   // F=1.5 > y=1.0
  assertClose("Pinball grad[2] (exact)", grad[2],  0.0);   // F=1.0 == y=1.0

  // hessians are tau*(1-tau) for pinball
  for i in 0..#n do
    assertClose("Pinball hess[" + i:string + "]", hess[i], tau * (1.0 - tau));

  // Test tau = 0.5 (median regression — symmetric)
  writeln("\n  tau=0.5 (median):");
  const tau2 = 0.5;
  var grad2 : [0..#n] real;
  var hess2 : [0..#n] real;
  var pb2 = new Pinball(tau=tau2);
  pb2.gradients(F, y, grad2, hess2);
  assertClose("Pinball(0.5) grad[0]", grad2[0], -0.5);
  assertClose("Pinball(0.5) grad[1]", grad2[1],  0.5);
}

// ------------------------------------------------------------------
// Smoke test: gradients computed on a synthetic distributed dataset
// Just checks shapes and that values are finite.
// ------------------------------------------------------------------
proc testDistributed() {
  writeln("\n--- Distributed smoke test ---");

  const data = makeSyntheticClassification(nSamples=1000, nFeatures=10);
  printDataSummary(data);

  var grad : [data.rowDom] real;
  var hess : [data.rowDom] real;

  // LogLoss on classification data
  var ll = new LogLoss();
  ll.gradients(data.F, data.y, grad, hess);

  // All gradients for F=0 (initial prediction) and y in {0,1}
  // should be either -0.5 (y=1) or +0.5 (y=0)
  const gradMin = min reduce grad;
  const gradMax = max reduce grad;
  writeln("  grad range: [", gradMin, ", ", gradMax, "]");
  if gradMin < -0.5 - EPS || gradMax > 0.5 + EPS then
    writeln("FAIL  LogLoss gradients out of expected range for zero-init F");
  else
    writeln("PASS  LogLoss gradient range for zero-init F");

  // Regression + pinball
  const reg = makeSyntheticRegression(nSamples=1000, nFeatures=10);
  var gp : [reg.rowDom] real;
  var hp : [reg.rowDom] real;
  var pb = new Pinball(tau=0.9);
  pb.gradients(reg.F, reg.y, gp, hp);

  // Pinball gradients are in {-tau, 0, 1-tau} = {-0.9, 0, 0.1}
  const gpMin = min reduce gp;
  const gpMax = max reduce gp;
  writeln("  Pinball grad range: [", gpMin, ", ", gpMax, "]");
  if gpMin < -0.9 - EPS || gpMax > 0.1 + EPS then
    writeln("FAIL  Pinball gradient range");
  else
    writeln("PASS  Pinball gradient range");
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" GBM Objective Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testMSE();
  testLogLoss();
  testPinball();
  testDistributed();

  writeln("\nDone.");
}
