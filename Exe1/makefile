frac-mpi: exe1.c
	rm -f out_julia.mp4 out_mandel.mp4
	rm -dfr ./images && mkdir ./images
	mpicc exe1.c -o exe1
	time mpirun -hostfile cluster exe1 3840 2160
#	ffmpeg -framerate 5 -pattern_type glob -i "./images/julia-*.ppm" -c:v libx264 -r 30 -pix_fmt yuv420p out_julia.mp4
#	ffmpeg -framerate 5 -pattern_type glob -i "./images/mandel-*.ppm" -c:v libx264 -r 30 -pix_fmt yuv420p out_mandel.mp4

clean: 
	rm -f exe1
	rm -f out_julia.mp4 out_mandel.mp4
	rm -dfr ./images && mkdir ./images
