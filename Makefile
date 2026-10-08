UV ?= uv

all: figures

figures:
	$(UV) run pplcog

sim:
	$(UV) run pplcog-sim

sim-quick:
	$(UV) run pplcog-sim --quick

test:
	$(UV) run pytest -q

clean:
	rm -rf out/*.pdf out/*.png out/*.json out/*.md out/*.tex out/*.csv out/models out/cache

.PHONY: all figures sim sim-quick test clean
