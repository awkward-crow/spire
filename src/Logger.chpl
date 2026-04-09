/*
  Logger.chpl
  -----------
  Minimal levelled logging for the GBM prototype.

  Log level is set at runtime via --logLevel=<level>:

    ./Booster --logLevel=INFO
    ./Booster --logLevel=TRACE

  Levels (ordered):
    NONE  — no output           (default)
    INFO  — per-tree summaries
    TRACE — per-node split details

  All output goes to stderr so it does not interfere with program
  stdout (e.g. CSV results from the comparison driver).
*/

module Logger {

  use IO;

  enum LogLevel { NONE, INFO, TRACE }

  config const logLevel: LogLevel = LogLevel.NONE;

  inline proc logInfo(msg: string) {
    if logLevel >= LogLevel.INFO then
      try! stderr.writeln("[INFO]  ", msg);
  }

  inline proc logTrace(msg: string) {
    if logLevel >= LogLevel.TRACE then
      try! stderr.writeln("[TRACE] ", msg);
  }

} // module Logger
