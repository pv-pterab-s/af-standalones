CPPFLAGS:=-Istubs

out/main: out/main.o
	mkdir -p $(dir $@)
	icpx -g -fsycl -fdiagnostics-absolute-paths out/main.o -Rno-debug-disables-optimization -o out/main

out/main.o: main.cpp fftconvolve_common.hpp fftconvolve_pack.hpp fftconvolve_pad.hpp fftconvolve_multiply.hpp fftconvolve_reorder.hpp
	mkdir -p $(dir $@)
	icpx -g -fsycl -fdiagnostics-absolute-paths $(CPPFLAGS) -Rno-debug-disables-optimization -c $< -o $@

clean:
	rm -rf out
