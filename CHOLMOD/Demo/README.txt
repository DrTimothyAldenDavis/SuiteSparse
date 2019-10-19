Demos for CHOLMOD

    cholmod_demo.c          a long demo
    cholmod_l_demo.c        same as cholmod_demo, but with long integers
    cholmod_demo.h          include file for cholmod*demo.c

    cholmod_simple.c        a very short and simple demo
    gpu.sh                  simple test for the GPU

    lperf.m                 test the performance of CHOLMOD in MATLAB
    Makefile                to compile the demos

    reade.f                 Fortran files to read sparse matrices
    readhb2.f                   in the Harwell/Boeing format
    readhb.f

    README.txt              this file

    Matrix                  folder

To compile and run the demos on the CPU, do "make" in this directory.
To run the demos on the GPU, you must first download the ND/ND6k matrix
from the SuiteSparse Matrix Collection, hosted at https://sparse.tamu.edu

Unpack the nd6k.mtx to your home directory.
Then do "make big" in this directory.  If you want to put the nd6k.mtx
file somewhere else, then simply edit the gpu.sh file.

