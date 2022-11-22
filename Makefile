test:
	nvcc -o a tests/main.cpp tests/test_kernel.cu -lfreeglut -lglew32
	make clean

clean:
	rm -f *.o
	rm -f *.exp
	rm -f *.lib
	rm -f *.pdb