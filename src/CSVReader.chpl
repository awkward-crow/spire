/*
  CSVReader.chpl
  --------------
  Loads a numeric CSV file into a GBMData record.

  Assumptions:
    - All data columns are parseable as real.
    - No missing values.
    - Optional single header row (skipped by default).
    - One column is the label; the rest become features (X).

  Algorithm:
    Pass 1 — scan for line count and column count; no line storage.
    Pass 2 — read floats directly from the channel into GBMData.
              reader.read(real) parses the next float literal (including
              scientific notation), skipping leading whitespace.  Commas
              are consumed explicitly with readByte().  Zero intermediate
              string allocation in the hot loop.

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

    // ---- Pass 1: count rows and columns (no line storage) -----------
    // Read line-by-line only to count; each line string is immediately
    // discarded.  Column count is inferred from the first relevant line.
    var nSamples  = 0;
    var nCols     = 0;
    {
      var f      = open(filename, ioMode.r);
      var reader = f.reader(locking=false);
      var line   : string;
      var firstLine = true;

      while reader.readLine(line) {
        const s = line.strip();
        if s == "" then continue;

        if firstLine {
          // Infer column count from the first line (header or data).
          nCols = s.count(",") + 1;
          firstLine = false;
          if hasHeader then continue;   // header: cols counted, not a data row
        }
        nSamples += 1;
      }
    }

    if nSamples == 0 then
      halt("readCSV: no data rows in '" + filename + "'");
    if nCols < 2 then
      halt("readCSV: need at least 2 columns, got " + nCols:string);

    const lCol      = if labelCol < 0 then nCols - 1 else labelCol;
    const nFeatures = nCols - 1;

    if lCol < 0 || lCol >= nCols then
      halt("readCSV: labelCol " + lCol:string
         + " out of range 0.." + (nCols-1):string);

    // ---- Allocate GBMData ------------------------------------------
    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    // ---- Pass 2: parse floats directly from channel ----------------
    // reader.read(real) handles scientific notation and skips leading
    // whitespace (including newlines between rows).  The comma between
    // fields is consumed by readByte(); the trailing newline on each
    // row is consumed as whitespace by the first read(real) of the
    // next row.
    {
      var f      = open(filename, ioMode.r);
      var reader = f.reader(locking=false);

      if hasHeader {
        var line: string;
        reader.readLine(line);   // discard header
      }

      for i in 0..#nSamples {
        var feat = 0;
        for col in 0..#nCols {
          var v: real;
          reader.read(v);
          if col == lCol then data.y[i] = v;
          else           { data.X[i, feat] = v; feat += 1; }
          if col < nCols - 1 then reader.readThrough(",");  // skip whitespace + ','
        }
        // trailing '\n' is consumed as whitespace by the next read(real)
      }
    }

    logInfo("readCSV: loaded " + nSamples:string + " rows x "
          + nFeatures:string + " features from '" + filename + "'");

    return data;
  }

} // module CSVReader
