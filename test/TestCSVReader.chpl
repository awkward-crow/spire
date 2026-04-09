/*
  TestCSVReader.chpl
  ------------------
  Tests for CSVReader.readCSV.

  Run:
    chpl TestCSVReader.chpl -M ../src -o build/TestCSVReader
    ./build/TestCSVReader
*/

use CSVReader;
use DataLayout;
use IO;
use Math;

proc assertTrue(name: string, cond: bool) {
  if !cond then writeln("FAIL  ", name);
  else       writeln("PASS  ", name);
}

proc assertClose(name: string, got: real, expected: real, tol: real = 1e-10) {
  if abs(got - expected) > tol then
    writeln("FAIL  ", name, " | got=", got, " expected=", expected);
  else
    writeln("PASS  ", name);
}

// Write a small CSV to a temp file and return the path.
proc writeTempCSV(path: string, header: bool) {
  var f = open(path, ioMode.cw);
  var w = f.writer(locking=false);
  if header then w.writeln("feat0,feat1,feat2,label");
  w.writeln("1.0,2.0,3.0,0.0");
  w.writeln("4.0,5.0,6.0,1.0");
  w.writeln("7.0,8.0,9.0,0.0");
  w.close();
  f.close();
}

// ------------------------------------------------------------------
// testBasicLoad
// ------------------------------------------------------------------
proc testBasicLoad() {
  writeln("\n--- basic load (last col = label) ---");

  const path = "/tmp/spire_test_basic.csv";
  writeTempCSV(path, header=true);

  const data = try! readCSV(path);

  assertTrue("numSamples = 3",  data.numSamples  == 3);
  assertTrue("numFeatures = 3", data.numFeatures == 3);

  assertClose("X[0,0]", data.X[0,0], 1.0);
  assertClose("X[0,1]", data.X[0,1], 2.0);
  assertClose("X[0,2]", data.X[0,2], 3.0);
  assertClose("y[0]",   data.y[0],   0.0);

  assertClose("X[1,0]", data.X[1,0], 4.0);
  assertClose("X[1,1]", data.X[1,1], 5.0);
  assertClose("X[1,2]", data.X[1,2], 6.0);
  assertClose("y[1]",   data.y[1],   1.0);

  assertClose("X[2,0]", data.X[2,0], 7.0);
  assertClose("y[2]",   data.y[2],   0.0);
}

// ------------------------------------------------------------------
// testNoHeader
// ------------------------------------------------------------------
proc testNoHeader() {
  writeln("\n--- no header ---");

  const path = "/tmp/spire_test_noheader.csv";
  writeTempCSV(path, header=false);

  const data = try! readCSV(path, hasHeader=false);

  assertTrue("numSamples = 3",  data.numSamples  == 3);
  assertTrue("numFeatures = 3", data.numFeatures == 3);
  assertClose("X[0,0] no header", data.X[0,0], 1.0);
  assertClose("y[0]   no header", data.y[0],   0.0);
}

// ------------------------------------------------------------------
// testLabelColFirst
// ------------------------------------------------------------------
proc testLabelColFirst() {
  writeln("\n--- label in first column ---");

  const path = "/tmp/spire_test_labelfirst.csv";
  {
    var f = open(path, ioMode.cw);
    var w = f.writer(locking=false);
    w.writeln("label,feat0,feat1");
    w.writeln("0.0,1.0,2.0");
    w.writeln("1.0,3.0,4.0");
    w.close();
    f.close();
  }

  const data = try! readCSV(path, labelCol=0);

  assertTrue("numSamples = 2",  data.numSamples  == 2);
  assertTrue("numFeatures = 2", data.numFeatures == 2);
  assertClose("y[0]   = 0.0", data.y[0],   0.0);
  assertClose("X[0,0] = 1.0", data.X[0,0], 1.0);
  assertClose("X[0,1] = 2.0", data.X[0,1], 2.0);
  assertClose("y[1]   = 1.0", data.y[1],   1.0);
  assertClose("X[1,0] = 3.0", data.X[1,0], 3.0);
}

// ------------------------------------------------------------------
// testWhitespace
// ------------------------------------------------------------------
proc testWhitespace() {
  writeln("\n--- whitespace around values ---");

  const path = "/tmp/spire_test_ws.csv";
  {
    var f = open(path, ioMode.cw);
    var w = f.writer(locking=false);
    w.writeln("feat0, feat1, label");
    w.writeln(" 1.5 , 2.5 , 1.0 ");
    w.close();
    f.close();
  }

  const data = try! readCSV(path);

  assertClose("X[0,0] with whitespace", data.X[0,0], 1.5);
  assertClose("X[0,1] with whitespace", data.X[0,1], 2.5);
  assertClose("y[0]   with whitespace", data.y[0],   1.0);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
proc main() {
  writeln("============================");
  writeln(" CSVReader Tests");
  writeln(" Locales: ", numLocales);
  writeln("============================");

  testBasicLoad();
  testNoHeader();
  testLabelColFirst();
  testWhitespace();

  writeln("\nDone.");
}
