/*
  Tree.chpl
  ---------
  Tree structure, node assignment, and prediction for one GBM tree.

  Tree structure: binary heap indexing throughout (training and prediction).
    Node n at depth d → heap index  2^d - 1 + n
    Left child  of heap index k →   2k + 1
    Right child of heap index k →   2k + 2
  Total nodes for maxDepth D:  2^(D+1) - 1

  nodeId[i] always stores an absolute heap index.  updateNodeAssign routes
  samples from heap index k to 2k+1 (left) or 2k+2 (right).  Leaf nodes
  leave nodeId unchanged — their samples never move to a child.

  Phantom nodes (children of depth-d leaves, never reached by any sample)
  accumulate zero gradient/hessian in the histogram, so findBestSplits
  returns valid=false for them.  finalizeLeaves records them as leaves with
  value=0; they are unreachable in applyTree.

  Prediction and node assignment both use the coforall + local-copy
  pattern: the small tree/split arrays are fetched to each locale once;
  the data-parallel loop over samples runs locally.
*/

module Tree {

  use DataLayout;
  use Histogram;
  use Splits;
  use Logger;

  // ------------------------------------------------------------------
  // FittedTree — stores one complete decision tree.
  // ------------------------------------------------------------------
  record FittedTree {
    var maxDepth : int;
    var nNodes   : int;
    var nodeDom  : domain(1);

    var feature  : [nodeDom] int;    // split feature  (internal nodes)
    var splitBin : [nodeDom] int;    // split bin       (internal nodes)
    var isLeaf   : [nodeDom] bool;
    var value    : [nodeDom] real;   // leaf prediction (leaf nodes)

    proc init() {
      this.maxDepth = 0;
      this.nNodes   = 1;
      this.nodeDom  = {0..#1};
    }

    proc init(maxDepth: int) {
      this.maxDepth = maxDepth;
      this.nNodes   = (1 << (maxDepth + 1)) - 1;   // 2^(maxDepth+1) - 1
      this.nodeDom  = {0..#nNodes};
    }
  }

  // Heap index for node n at depth d.
  inline proc heapIdx(depth: int, n: int): int {
    return (1 << depth) - 1 + n;   // 2^depth - 1 + n
  }

  // ------------------------------------------------------------------
  // updateNodeAssign
  //
  // Routes each sample left or right using absolute heap indexing:
  //   left child  of heap index k → 2k + 1
  //   right child of heap index k → 2k + 2
  // Samples in leaf nodes (splits[k].valid = false) keep their nodeId.
  //
  // splits lives on locale 0; fetched to each locale before the forall.
  // ------------------------------------------------------------------
  proc updateNodeAssign(
      data       : GBMData,
      splits     : [] SplitInfo,
      ref nodeId : [] int
  ) {
    coforall loc in Locales with (ref nodeId) {
      on loc {
        const localSplits = splits;
        const localDom    = data.rowDom.localSubdomain();
        forall i in localDom with (ref nodeId) {
          const k = nodeId[i];
          if localSplits[k].valid {
            if data.Xb[i, localSplits[k].feature] <= localSplits[k].bin then
              nodeId[i] = 2 * k + 1;
            else
              nodeId[i] = 2 * k + 2;
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // recordLevel
  //
  // Stores one depth level of results into the FittedTree.
  // Both splits and histogram are indexed by absolute heap index.
  //
  // Split nodes:   record feature/bin as internal.
  // Unsplit nodes: compute leaf value from histogram and mark as leaf.
  //
  // Called on locale 0 after findBestSplits for each depth.
  // ------------------------------------------------------------------
  proc recordLevel(
      ref tree  : FittedTree,
      splits    : [] SplitInfo,
      hist      : HistogramData,
      depth     : int,
      lambda    : real,
      eta       : real
  ) {
    for n in 0..#(1 << depth) {
      const idx = heapIdx(depth, n);
      if splits[idx].valid {
        tree.feature[idx]  = splits[idx].feature;
        tree.splitBin[idx] = splits[idx].bin;
        tree.isLeaf[idx]   = false;
      } else {
        const G = + reduce hist.grad[idx, 0, ..];
        const H = + reduce hist.hess[idx, 0, ..];
        tree.isLeaf[idx] = true;
        tree.value[idx]  = eta * leafValue(G, H, lambda);
        logTrace("recordLevel: early leaf node=" + idx:string
               + " depth="                      + depth:string
               + " G="                          + G:string
               + " H="                          + H:string
               + " value="                      + tree.value[idx]:string);
      }
    }
  }

  // ------------------------------------------------------------------
  // finalizeLeaves
  //
  // At maxDepth all remaining un-marked nodes become leaves.
  // Phantom nodes (no samples) get value=0 and are unreachable in
  // prediction.  Called once after the final histogram is built.
  // ------------------------------------------------------------------
  proc finalizeLeaves(
      ref tree  : FittedTree,
      hist      : HistogramData,
      depth     : int,
      lambda    : real,
      eta       : real
  ) {
    for n in 0..#(1 << depth) {
      const idx = heapIdx(depth, n);
      if !tree.isLeaf[idx] {
        const G = + reduce hist.grad[idx, 0, ..];
        const H = + reduce hist.hess[idx, 0, ..];
        tree.isLeaf[idx] = true;
        tree.value[idx]  = eta * leafValue(G, H, lambda);
        logTrace("finalizeLeaves: leaf node=" + idx:string
               + " G="                       + G:string
               + " H="                       + H:string
               + " value="                   + tree.value[idx]:string);
      }
    }
  }

  // ------------------------------------------------------------------
  // applyTree
  //
  // F[i] += tree.value[leaf] for each sample.
  //
  // eta is already baked into tree.value at recordLevel/finalizeLeaves
  // time; it is NOT applied again here.
  //
  // Tree arrays are fetched to each locale once before the forall to
  // avoid per-sample remote GETs while walking the tree.
  // ------------------------------------------------------------------
  proc applyTree(
      data    : GBMData,
      tree    : FittedTree,
      ref F   : [] real
  ) {
    coforall loc in Locales with (ref F) {
      on loc {
        const localFeature  = tree.feature;
        const localSplitBin = tree.splitBin;
        const localIsLeaf   = tree.isLeaf;
        const localValue    = tree.value;

        const localDom = data.rowDom.localSubdomain();
        forall i in localDom with (ref F) {
          var node = 0;
          while !localIsLeaf[node] {
            if data.Xb[i, localFeature[node]] <= localSplitBin[node] then
              node = 2 * node + 1;
            else
              node = 2 * node + 2;
          }
          F[i] += localValue[node];
        }
      }
    }
  }

} // module Tree
