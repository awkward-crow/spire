/*
  TestTree.chpl
  -------------
  Tests for Tree.updateNodeAssign and Tree.applyTree.

  Run:
    chpl TestTree.chpl -M ../src -o TestTree
    ./TestTree
*/

use Tree;
use Splits;
use Histogram;
use Binning;
use DataLayout;
use SyntheticData;
use Objectives;
use Booster;

const EPS = 1e-10;

proc assertClose(name: string, got: real, expected: real, tol: real = EPS) {
  if abs(got - expected) > tol then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

proc assertEq(name: string, got: int, expected: int) {
  if got != expected then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

// ------------------------------------------------------------------
// testUpdateNodeAssign
//
// 4 samples, 1 feature.  Split root (node 0): feature=0, bin=5.
// Xb values: [2, 5, 8, 3].  left=1, right=2.
// Expected nodeId: [1, 1, 2, 1]
// ------------------------------------------------------------------
proc testUpdateNodeAssign() {
  writeln("\n--- updateNodeAssign ---");

  var data = new GBMData(numSamples=4, numFeatures=1);
  data.Xb[0, 0] = 2: uint(8);
  data.Xb[1, 0] = 5: uint(8);
  data.Xb[2, 0] = 8: uint(8);
  data.Xb[3, 0] = 3: uint(8);

  var split = new SplitInfo();
  split.feature = 0;
  split.bin     = 5;
  split.valid   = true;

  var nodeId: [data.rowDom] int = 0;
  updateNodeAssign(data, 0, split, 1, 2, nodeId);

  assertEq("sample 0 (Xb=2 <= 5 → left,  node 1)", nodeId[0], 1);
  assertEq("sample 1 (Xb=5 <= 5 → left,  node 1)", nodeId[1], 1);
  assertEq("sample 2 (Xb=8 >  5 → right, node 2)", nodeId[2], 2);
  assertEq("sample 3 (Xb=3 <= 5 → left,  node 1)", nodeId[3], 1);
}

// ------------------------------------------------------------------
// testApplyTree
//
// Manually build a depth-1 tree with child pointers:
//   node 0 (root): split feature=0, bin=5, left=1, right=2
//   node 1 (left):  leaf, value= 1.0
//   node 2 (right): leaf, value=-1.0
//
// Same 4 samples.  Expected F: [1.0, 1.0, -1.0, 1.0]
// ------------------------------------------------------------------
proc testApplyTree() {
  writeln("\n--- applyTree ---");

  var data = new GBMData(numSamples=4, numFeatures=1);
  data.Xb[0, 0] = 2: uint(8);
  data.Xb[1, 0] = 5: uint(8);
  data.Xb[2, 0] = 8: uint(8);
  data.Xb[3, 0] = 3: uint(8);

  var tree = new FittedTree(numLeaves=2);   // capacity = 3 nodes
  tree.nNodes         = 3;
  tree.isLeaf[0]      = false;
  tree.feature[0]     = 0;
  tree.splitBin[0]    = 5;
  tree.leftChild[0]   = 1;
  tree.rightChild[0]  = 2;
  tree.isLeaf[1]      = true;
  tree.value[1]       = 1.0;
  tree.isLeaf[2]      = true;
  tree.value[2]       = -1.0;

  var F: [data.rowDom] real = 0.0;
  applyTree(data, tree, F);

  assertClose("F[0] (Xb=2 → left)",  F[0],  1.0);
  assertClose("F[1] (Xb=5 → left)",  F[1],  1.0);
  assertClose("F[2] (Xb=8 → right)", F[2], -1.0);
  assertClose("F[3] (Xb=3 → left)",  F[3],  1.0);
}

// ------------------------------------------------------------------
// testEndToEnd
//
// One full boost() call on synthetic regression data.
// Checks that loss decreases and F is updated.
// ------------------------------------------------------------------
proc testEndToEnd() {
  writeln("\n--- end-to-end (boost + predict) ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=5);
  computeBins(data);

  const lossBefore = mseLoss(data.F, data.y);

  var cfg = new BoosterConfig(nTrees=5, numLeaves=4, eta=0.3, lambda=1.0);
  boost(data, new MSE(), cfg);

  const lossAfter = mseLoss(data.F, data.y);

  assertTrue("MSE decreases after boosting",   lossAfter < lossBefore);
  assertTrue("F is non-zero after boosting",   (max reduce abs(data.F)) > 0.0);

  var allFinite = true;
  for i in data.rowDom do
    if !isFinite(data.F[i]) { allFinite = false; break; }
  assertTrue("all F values are finite", allFinite);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Tree Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testUpdateNodeAssign();
  testApplyTree();
  testEndToEnd();

  writeln("\nDone.");
}
