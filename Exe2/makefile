frac-mpi: exe2.c
	rm -f out_julia.mp4 out_mandel.mp4
	rm -f mandel.ppm julia.ppm
	rm -dfr ./images && mkdir ./images
	mpicc exe2.c -o exe2
	mpirun -hostfile cluster exe2 3840 2160 100 0.6
	ffmpeg -framerate 5 -pattern_type glob -i "./images/Mandel_diff-*.ppm" -c:v libx264 -r 30 -pix_fmt yuv420p out_mandel.mp4
	ffmpeg -framerate 5 -pattern_type glob -i "./images/Julia_diff-*.ppm" -c:v libx264 -r 30 -pix_fmt yuv420p out_julia.mp4

clean: 
	rm -f exe2
	rm -f out_julia.mp4 out_mandel.mp4
	rm -dfr ./images