CHPL      = chpl
CHPLFLAGS = -M ../src
BUILD     = build

TESTS = TestObjectives TestBinning

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
