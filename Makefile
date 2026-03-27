profile-cuda:
	@scripts/profile_cuda.sh

clean-profile-cuda:
	rm *.nsys-rep *.sqlite

plot-profile:
	@scripts/plot_profile.sh
