# Makefile for rupantara — pure-Cyrius transformer-forward library.
#
#   make build   — link-check the library (programs/smoke.cyr)
#   make test    — CPU test suites (tests/tcyr/*.tcyr; the golden gate among them)
#   make test-aarch64 — every suite cross-built for aarch64, run under qemu-aarch64
#   make golden-bless — rewrite tests/golden/*.txt after a reviewed numerical change
#   make fuzz    — seeded property fuzzing (tests/rupantara.fcyr; exit 1 = a property failed)
#   make bench   — benchmarks (tests/rupantara.bcyr); scripts/bench-history.sh records them
#   make dist    — regenerate dist/rupantara.cyr via `cyrius distlib`
#   make lint / fmt-check / vet / doc  — quality gates
#   make clean   — scrub build/

CYRIUS ?= cyrius

# every Cyrius source the gates cover: the library, programs, suites, shared test
# helpers / oracles, the benchmark and the fuzz harness
SOURCES = src/*.cyr programs/*.cyr tests/tcyr/*.tcyr tests/*.cyr tests/*.bcyr tests/*.fcyr

.PHONY: check-lib-wiring
check-lib-wiring:
	@if [ -L lib ]; then \
		echo "ERROR: lib/ is a symlink ($$(readlink lib)). Fix: rm lib && cyrius deps"; \
		exit 1; \
	fi

.PHONY: build
build: check-lib-wiring
	@mkdir -p build
	CYRIUS_DCE=1 $(CYRIUS) build programs/smoke.cyr build/rupantara_smoke
	@echo "smoke: $$(wc -c < build/rupantara_smoke) bytes"

.PHONY: test
test: check-lib-wiring
	@for f in tests/tcyr/*.tcyr; do $(CYRIUS) test "$$f" || exit 1; done

# every suite cross-built for aarch64 and run under qemu-aarch64 (CI's aarch64 job)
.PHONY: test-aarch64
test-aarch64: check-lib-wiring
	@mkdir -p build
	@for f in tests/tcyr/*.tcyr; do \
		n=$$(basename $$f .tcyr); \
		$(CYRIUS) build --aarch64 $$f build/$${n}_a64 > /dev/null || exit 1; \
		echo "== $$n"; qemu-aarch64 build/$${n}_a64 || exit 1; \
	done

# Rewrite tests/golden/{x86_64,aarch64}.txt from the current tree. Only after review:
# a moved bit is a numerical change (ADR 0002), so name it in the CHANGELOG, and check
# `git diff tests/golden/` shows only the values you meant to move.
.PHONY: golden-bless
golden-bless: check-lib-wiring
	RU_GOLDEN_BLESS=1 $(CYRIUS) test tests/tcyr/golden.tcyr
	RU_GOLDEN_BLESS=1 $(CYRIUS) test --aarch64 tests/tcyr/golden.tcyr

.PHONY: fuzz
fuzz: check-lib-wiring
	@mkdir -p build
	CYRIUS_DCE=1 $(CYRIUS) build tests/rupantara.fcyr build/rupantara-fuzz
	timeout 120 ./build/rupantara-fuzz

.PHONY: bench
bench: check-lib-wiring
	CYRIUS_DCE=1 $(CYRIUS) bench tests/rupantara.bcyr

.PHONY: doc
doc:
	$(CYRIUS) doc --check src/*.cyr

.PHONY: lint
lint:
	@fail=0; \
	for f in $(SOURCES); do \
		out=$$($(CYRIUS) lint $$f 2>&1); echo "$$out"; \
		echo "$$out" | grep -qE '^\s*warn ' && fail=1; \
	done; \
	[ $$fail -eq 0 ] || { echo "lint: warnings present"; exit 1; }

.PHONY: fmt-check
fmt-check:
	@fail=0; \
	for f in $(SOURCES); do \
		if ! $(CYRIUS) fmt $$f --check > /dev/null 2>&1; then \
			echo "needs fmt: $$f"; fail=1; \
		fi; \
	done; \
	[ $$fail -eq 0 ] || { echo "fmt: drift detected"; exit 1; }

.PHONY: vet
vet:
	$(CYRIUS) vet programs/smoke.cyr

.PHONY: dist
dist:
	$(CYRIUS) distlib

.PHONY: test-all
test-all: dist test fuzz

.PHONY: clean
clean:
	rm -rf build/
