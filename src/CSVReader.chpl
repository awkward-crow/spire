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
    Pass 1 — read all data lines into a local list.
    Pass 2 — parse and fill GBMData.X and GBMData.y.

  GBMData requires numSamples and numFeatures at construction, so the
  list buffer avoids needing a separate counting pass over the file.

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
  use List;
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

    // ---- Pass 1: buffer all data lines ------------------------------
    var lines: list(string);
    {
      var f      = open(filename, ioMode.r);
      var reader = f.reader(locking=false);
      var skip   = hasHeader;
      for line in reader.lines() {
        const s = line.strip();
        if s == "" then continue;
        if skip { skip = false; continue; }
        lines.pushBack(s);
      }
    }

    if lines.size == 0 then
      halt("readCSV: no data rows in '" + filename + "'");

    // ---- Infer dimensions from first data row -----------------------
    const firstFields = lines[0].split(",");
    const nCols       = firstFields.size;
    const lCol        = if labelCol < 0 then nCols - 1 else labelCol;
    const nFeatures   = nCols - 1;

    if nFeatures < 1 then
      halt("readCSV: need at least 2 columns, got " + nCols:string);
    if lCol < 0 || lCol >= nCols then
      halt("readCSV: labelCol " + lCol:string
         + " out of range 0.." + (nCols-1):string);

    // ---- Pass 2: parse rows into GBMData ----------------------------
    const nSamples = lines.size;
    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    for i in 0..#nSamples {
      const fields = lines[i].split(",");
      if fields.size != nCols then
        halt("readCSV: row " + i:string + " has " + fields.size:string
           + " columns, expected " + nCols:string);
      var feat = 0;
      for col in 0..#nCols {
        const v = fields[col].strip(): real;
        if col == lCol then data.y[i] = v;
        else { data.X[i, feat] = v; feat += 1; }
      }
    }

    logInfo("readCSV: loaded " + nSamples:string + " rows x "
          + nFeatures:string + " features from '" + filename + "'");

    return data;
  }

} // module CSVReader
