Demos for CHOLMOD

    cholmod_di_demo.c       double/int32 demo
    cholmod_dl_demo.c       double/int64 demo
    cholmod_si_demo.c       float/int32 demo
    cholmod_sl_demo.c       float/int64 demo

    cholmod_demo.h          include file for cholmod*demo.c

    cholmod_simple.c        a very short and simple demo (double precision)
    cholmod_s_simple.c      single precision version of cholmod_simple.c

    gpu.sh                  simple test for the GPU

    lperf.m                 test the performance of CHOLMOD in MATLAB
    Makefile                to compile the demos

    reade.f                 Fortran files to read sparse matrices
    readhb2.f                   in the Harwell/Boeing format
    readhb.f

    README.txt              this file

    Matrix                  folder with test matrices

To compile and run the demos on the CPU, do "make demos" in the parent
directory (SuiteSparse/CHOLMOD).

To run the demos on the GPU, you must first download the ND/ND6k matrix
from the SuiteSparse Matrix Collection, hosted at https://sparse.tamu.edu

Unpack the nd6k.mtx to your home directory.
Then do "./gpu.sh" in this directory.  If you want to put the nd6k.mtx
file somewhere else, then simply edit the gpu.sh file.

