/*
  TestTree.chpl
  -------------
  Tests for Tree.heapIdx, updateNodeAssign, recordLevel,
  finalizeLeaves, and applyTree.

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
// testHeapIdx
// ------------------------------------------------------------------
proc testHeapIdx() {
  writeln("\n--- heapIdx ---");

  assertEq("depth=0 n=0 (root)",          heapIdx(0, 0), 0);
  assertEq("depth=1 n=0 (left of root)",  heapIdx(1, 0), 1);
  assertEq("depth=1 n=1 (right of root)", heapIdx(1, 1), 2);
  assertEq("depth=2 n=0",                 heapIdx(2, 0), 3);
  assertEq("depth=2 n=1",                 heapIdx(2, 1), 4);
  assertEq("depth=2 n=2",                 heapIdx(2, 2), 5);
  assertEq("depth=2 n=3",                 heapIdx(2, 3), 6);
}

// ------------------------------------------------------------------
// testUpdateNodeAssign
//
// 4 samples, 1 feature.  Split at root (heap idx 0): feature=0, bin=5.
// Xb values: [2, 5, 8, 3]
//
// Absolute heap indexing: left = 2*0+1 = 1, right = 2*0+2 = 2.
// Expected nodeId after: [1, 1, 2, 1]
// ------------------------------------------------------------------
proc testUpdateNodeAssign() {
  writeln("\n--- updateNodeAssign ---");

  var data = new GBMData(numSamples=4, numFeatures=1);
  data.Xb[0, 0] = 2: uint(8);
  data.Xb[1, 0] = 5: uint(8);
  data.Xb[2, 0] = 8: uint(8);
  data.Xb[3, 0] = 3: uint(8);

  // splits indexed by heap index; root is at index 0
  var splits: [0..2] SplitInfo;
  splits[0].feature = 0;
  splits[0].bin     = 5;
  splits[0].valid   = true;

  var nodeId: [data.rowDom] int = 0;
  updateNodeAssign(data, splits, nodeId);

  assertEq("sample 0 (Xb=2 <= 5 → left,  heap 1)", nodeId[0], 1);
  assertEq("sample 1 (Xb=5 <= 5 → left,  heap 1)", nodeId[1], 1);
  assertEq("sample 2 (Xb=8 >  5 → right, heap 2)", nodeId[2], 2);
  assertEq("sample 3 (Xb=3 <= 5 → left,  heap 1)", nodeId[3], 1);
}

// ------------------------------------------------------------------
// testApplyTree
//
// Manually build a depth-1 tree:
//   root (idx 0): split feature=0, bin=5
//   left (idx 1): leaf, value= 1.0
//   right(idx 2): leaf, value=-1.0
//
// Same 4 samples.  eta=1.0.
// Expected F: [1.0, 1.0, -1.0, 1.0]
// ------------------------------------------------------------------
proc testApplyTree() {
  writeln("\n--- applyTree ---");

  var data = new GBMData(numSamples=4, numFeatures=1);
  data.Xb[0, 0] = 2: uint(8);
  data.Xb[1, 0] = 5: uint(8);
  data.Xb[2, 0] = 8: uint(8);
  data.Xb[3, 0] = 3: uint(8);

  var tree = new FittedTree(maxDepth=1);
  tree.isLeaf[0]   = false;
  tree.feature[0]  = 0;
  tree.splitBin[0] = 5;
  tree.isLeaf[1]   = true;
  tree.value[1]    = 1.0;
  tree.isLeaf[2]   = true;
  tree.value[2]    = -1.0;

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
// One full boosting round on synthetic regression data using absolute
// heap indexing throughout.  Histogram sized to tree.nNodes.
//
// Checks:
//   - Root has a valid split
//   - F is updated (non-zero)
//   - All F values are finite
//   - Gradient sum moves closer to zero after the tree is applied
// ------------------------------------------------------------------
proc testEndToEnd() {
  writeln("\n--- end-to-end (one boosting round) ---");

  var data = makeSyntheticRegression(nSamples=500, nFeatures=5);
  computeBins(data);
  computeGradients(Objective.MSE, data.F, data.y, data.grad, data.hess);

  const gradSumBefore = + reduce data.grad;

  var tree   = new FittedTree(maxDepth=1);
  var nodeId : [data.rowDom] int = 0;
  var hist   = new HistogramData(maxNodes=tree.nNodes, nFeatures=data.numFeatures);

  // Depth 0
  buildHistograms(data, nodeId, hist);
  const splits0 = findBestSplits(hist, lambda=1.0, minHess=1.0);
  assertTrue("root (heap 0) has a valid split", splits0[0].valid);
  recordLevel(tree, splits0, hist, depth=0, lambda=1.0, eta=0.1);
  updateNodeAssign(data, splits0, nodeId);

  // Depth 1 — finalize leaves
  buildHistograms(data, nodeId, hist);
  finalizeLeaves(tree, hist, depth=1, lambda=1.0, eta=0.1);

  applyTree(data, tree, data.F);

  assertTrue("F is updated after applyTree", (max reduce abs(data.F)) > 0.0);

  var allFinite = true;
  for i in data.rowDom do
    if !isFinite(data.F[i]) { allFinite = false; break; }
  assertTrue("all F values are finite", allFinite);

  computeGradients(Objective.MSE, data.F, data.y, data.grad, data.hess);
  const gradSumAfter = + reduce data.grad;
  assertTrue("gradient sum closer to zero after one tree",
             abs(gradSumAfter) <= abs(gradSumBefore) + EPS);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" Tree Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testHeapIdx();
  testUpdateNodeAssign();
  testApplyTree();
  testEndToEnd();

  writeln("\nDone.");
}
