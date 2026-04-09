CHPL      = chpl
CHPLFLAGS = -M ../src
BUILD     = build

# Optimize for the build machine's CPU (enables AVX2/AVX-512 auto-vectorization).
# Override for heterogeneous clusters where node CPUs differ, e.g.:
#   make CHPL_TARGET_CPU=broadwell
export CHPL_TARGET_CPU ?= native

TESTS = TestObjectives TestBinning TestHistogram TestSplits TestTree TestBooster TestPredict

.PHONY: all run clean $(TESTS)

all: $(TESTS)

$(BUILD):
	mkdir -p $(BUILD)

$(TESTS): %: %.chpl | $(BUILD)
	$(CHPL) $(CHPLFLAGS) $< -o $(BUILD)/$@

run: all
	$(foreach t, $(TESTS), $(BUILD)/$(t);)

clean:
	rm -rf $(BUILD)
