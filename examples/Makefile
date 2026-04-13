CHPL      = chpl
CHPLFLAGS = -M ../src --fast

# Build without optimisations (safety checks, bounds checking):
#   make DEBUG=1 BreastCancer
# Build with debug symbols for profiling (implies DEBUG):
#   make PROFILE=1 BreastCancer
ifdef DEBUG
CHPLFLAGS := -M ../src
endif
ifdef PROFILE
CHPLFLAGS := -M ../src --fast -g
endif
BUILD     = build

export CHPL_TARGET_CPU ?= native

PROGRAMS = CaliforniaHousing Bicycle BreastCancer CoverType SUSY

$(BUILD):
	mkdir -p $(BUILD)

$(PROGRAMS): %: %.chpl | $(BUILD)
	$(CHPL) $(CHPLFLAGS) $< -o $(BUILD)/$@

clean:
	rm -rf $(BUILD)
