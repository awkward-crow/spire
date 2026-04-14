/*
  Tree.chpl
  ---------
  Tree structure, node assignment, and prediction for one GBM tree.

  Tree structure: explicit child pointers (leaf-wise growth).

  Nodes are allocated sequentially as the tree grows.  Root is always
  index 0.  Each split consumes two new indices (left, right child).
  Maximum nodes for numLeaves leaves: 2*numLeaves - 1.

  nodeId[i] is the node index currently holding sample i.
  updateNodeAssign routes samples at a split node to its two children.

  applyTree walks each sample from root to leaf following child pointers.
  Tree arrays are fetched to each locale once to avoid per-sample remote GETs.
*/

module Tree {

  use DataLayout;
  use Splits;

  // ------------------------------------------------------------------
  // FittedTree
  //
  // capacity  — pre-allocated node slots (2*numLeaves - 1)
  // nNodes    — nodes in use; starts at 1 (root)
  //
  // Internal nodes: isLeaf=false, leftChild/rightChild set.
  // Leaf nodes:     isLeaf=true,  value set.
  // ------------------------------------------------------------------
  record FittedTree {
    var capacity : int;
    var nNodes   : int;
    var nodeDom  : domain(1);

    var leftChild  : [nodeDom] int;
    var rightChild : [nodeDom] int;
    var feature    : [nodeDom] int;
    var splitBin   : [nodeDom] int;
    var isLeaf     : [nodeDom] bool;
    var value      : [nodeDom] real;

    proc init() {
      this.capacity = 1;
      this.nNodes   = 1;
      this.nodeDom  = {0..#1};
    }

    proc init(numLeaves: int) {
      const cap     = 2 * numLeaves - 1;
      this.capacity = cap;
      this.nNodes   = 1;
      this.nodeDom  = {0..#cap};
    }
  }

  // ------------------------------------------------------------------
  // updateNodeAssign
  //
  // Routes samples currently at splitNode to leftChild or rightChild.
  // ------------------------------------------------------------------
  proc updateNodeAssign(
      data       : GBMData,
      splitNode  : int,
      split      : SplitInfo,
      leftChild  : int,
      rightChild : int,
      ref nodeId : [] int
  ) {
    coforall loc in Locales with (ref nodeId) {
      on loc {
        const localDom = data.rowDom.localSubdomain();
        forall i in localDom with (ref nodeId) {
          if nodeId[i] == splitNode {
            if data.Xb[i, split.feature] <= split.bin then
              nodeId[i] = leftChild;
            else
              nodeId[i] = rightChild;
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // applyTree
  //
  // F[i] += tree.value[leaf] for each sample.
  // eta is already baked into tree.value — not applied again here.
  // Tree arrays fetched to each locale once before the forall.
  // ------------------------------------------------------------------
  proc applyTree(
      data  : GBMData,
      tree  : FittedTree,
      ref F : [] real
  ) {
    coforall loc in Locales with (ref F) {
      on loc {
        const localLeftChild  = tree.leftChild;
        const localRightChild = tree.rightChild;
        const localFeature    = tree.feature;
        const localSplitBin   = tree.splitBin;
        const localIsLeaf     = tree.isLeaf;
        const localValue      = tree.value;

        const localDom = data.rowDom.localSubdomain();
        forall i in localDom with (ref F) {
          var node = 0;
          while !localIsLeaf[node] {
            if data.Xb[i, localFeature[node]] <= localSplitBin[node] then
              node = localLeftChild[node];
            else
              node = localRightChild[node];
          }
          F[i] += localValue[node];
        }
      }
    }
  }

} // module Tree
