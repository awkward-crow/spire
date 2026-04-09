CHPL    = chpl
CHPLFLAGS = -M ../src

TESTS = TestObjectives

.PHONY: all run clean $(TESTS)

all: $(TESTS)

$(TESTS): %: %.chpl
	$(CHPL) $(CHPLFLAGS) $< -o $@

run: all
	$(foreach t, $(TESTS), ./$(t);)

clean:
	rm -f $(TESTS)
