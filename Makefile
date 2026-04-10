nsys-profile:
	@scripts/nsys_profile.sh

clean-nsys-profile:
	-rm *.nsys-rep *.sqlite

plot-profile:
	@scripts/plot_profile.sh

plot-residuals:
	@scripts/plot_residuals.sh

run-all:
	@scripts/run_all.sh $(ARGS)

.PHONY: benchmark
benchmark:
	@scripts/benchmark.sh
