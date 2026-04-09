CHPL      = chpl
CHPLFLAGS = -M ../src
BUILD     = build

export CHPL_TARGET_CPU ?= native

PROGRAMS = CaliforniaHousing

$(BUILD):
	mkdir -p $(BUILD)

$(PROGRAMS): %: %.chpl | $(BUILD)
	$(CHPL) $(CHPLFLAGS) $< -o $(BUILD)/$@

clean:
	rm -rf $(BUILD)
