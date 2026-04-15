/*
  CSVReader.chpl
  --------------
  Loads a numeric CSV file into a GBMData record.

  Assumptions:
    - All data columns are parseable as real.
    - No missing values.
    - Optional single header row (skipped by default).
    - One column is the label; the rest become features (X).

  Algorithm (parallel):
    Step 1 — read first data line to infer column count.
    Step 2 — divide file into here.maxTaskPar byte-range chunks; find the
              newline boundary that aligns each chunk to a whole row.
    Step 3 — parallel pass 1: each task counts its rows via readLine.
    Step 4 — prefix-sum row counts; allocate GBMData once.
    Step 5 — parallel pass 2: each task rereads its chunk, parsing floats
              directly with reader.read(real) and readThrough(",").

  Feature column ordering: columns are mapped to feature indices 0..
  left-to-right, skipping labelCol.  E.g. with labelCol=2 and 4 cols:
    col 0 → feature 0
    col 1 → feature 1
    col 2 → label
    col 3 → feature 2
*/

module CSVReader {

  use DataLayout;
  use IO;
  use FileSystem;
  use Logger;

  // ------------------------------------------------------------------
  // readCSV
  //
  // filename  — path to the CSV file
  // labelCol  — column index of the label (-1 = last column)
  // hasHeader — if true, the first row is treated as a header and skipped
  // ------------------------------------------------------------------
  proc readCSV(
      filename  : string,
      labelCol  : int  = -1,
      hasHeader : bool = true
  ): GBMData throws {

    // ---- Step 1: infer column count from first data line ------------
    var nCols = 0;
    {
      var f      = open(filename, ioMode.r);
      var reader = f.reader(locking=false);
      var line   : string;
      if hasHeader then reader.readLine(line);  // skip header
      if reader.readLine(line) {
        const s = line.strip();
        nCols = s.count(",") + 1;
      }
    }

    if nCols < 2 then
      halt("readCSV: need at least 2 columns, got " + nCols:string);

    const lCol      = if labelCol < 0 then nCols - 1 else labelCol;
    const nFeatures = nCols - 1;

    if lCol < 0 || lCol >= nCols then
      halt("readCSV: labelCol " + lCol:string
         + " out of range 0.." + (nCols-1):string);

    // ---- Step 2: find newline-aligned chunk boundaries --------------
    // We seek to rawStart = tid * (fsize / nTasks) then scan forward
    // until the next newline.  This is O(nTasks × avgLineLen) bytes of
    // sequential IO — negligible compared to the full file read.
    const fsize   = getFileSize(filename);
    const nTasks  = here.maxTaskPar;
    const rawChunk = fsize / nTasks;

    // chunkBounds[tid] = byte offset where task tid begins reading.
    // chunkBounds[nTasks] = fsize (sentinel for the last task).
    var chunkBounds: [0..nTasks] int;
    chunkBounds[0]      = 0;
    chunkBounds[nTasks] = fsize;

    for tid in 1..#(nTasks - 1) {
      const rawStart = tid * rawChunk;
      var f = open(filename, ioMode.r);
      var r = f.reader(locking=false, region=rawStart..);
      var b: uint(8);
      while r.readByte(b) && b != 10 {}   // scan to (and consume) newline
      chunkBounds[tid] = r.offset();       // absolute position after newline
    }

    // Task 0: if there is a header, advance past it.
    if hasHeader {
      var f = open(filename, ioMode.r);
      var r = f.reader(locking=false);
      var line: string;
      r.readLine(line);
      chunkBounds[0] = r.offset();
    }

    // ---- Step 3: parallel row counting ------------------------------
    var rowCounts: [0..#nTasks] int;

    coforall tid in 0..#nTasks with (ref rowCounts) {
      const start = chunkBounds[tid];
      const end_  = chunkBounds[tid + 1];
      var f = open(filename, ioMode.r);
      var r = f.reader(locking=false, region=start..<end_);
      var line: string;
      while r.readLine(line) { rowCounts[tid] += 1; }
    }

    const nSamples = + reduce rowCounts;
    if nSamples == 0 then
      halt("readCSV: no data rows in '" + filename + "'");

    // ---- Prefix-sum row offsets -------------------------------------
    var rowOffsets: [0..#nTasks] int;
    rowOffsets[0] = 0;
    for tid in 1..#(nTasks - 1) do
      rowOffsets[tid] = rowOffsets[tid - 1] + rowCounts[tid - 1];

    // ---- Step 4: allocate GBMData -----------------------------------
    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    // ---- Step 5: parallel float parsing -----------------------------
    // Each task reopens the file at its chunk start and parses exactly
    // rowCounts[tid] rows, writing into its slice of data.X / data.y.
    // Writes are to disjoint row ranges so no locking is required.
    coforall tid in 0..#nTasks with (ref data) {
      const start  = chunkBounds[tid];
      const nRows  = rowCounts[tid];
      const rowOff = rowOffsets[tid];

      var f = open(filename, ioMode.r);
      var r = f.reader(locking=false, region=start..);

      for i in 0..#nRows {
        const row = rowOff + i;
        var feat  = 0;
        for col in 0..#nCols {
          var v: real;
          r.read(v);
          if col == lCol then data.y[row] = v: real(32);
          else           { data.X[row, feat] = v; feat += 1; }
          if col < nCols - 1 then r.readThrough(",");
        }
      }
    }

    logInfo("readCSV: loaded " + nSamples:string + " rows x "
          + nFeatures:string + " features from '" + filename + "'"
          + " (" + nTasks:string + " tasks)");

    return data;
  }

} // module CSVReader
