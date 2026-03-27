nsys-profile:
	@scripts/nsys_profile.sh

clean-nsys-profile:
	-rm *.nsys-rep *.sqlite

plot-profile:
	@scripts/plot_profile.sh

profile:
	@scripts/profile.sh
