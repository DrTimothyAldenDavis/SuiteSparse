Results on a Dell XPS 13 9380 laptop.

Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (up to 4.6GHz),
16GB of RAM.  MATLAB R2020a.  Ubuntu 18.04.  GraphBLAS and
the mexFunction interface compiled with gcc 7.5.0.  The CPU
has 4 hardware cores, and 4 threads were used (the default
when running OpenMP inside MATLAB).

Note that MATLAB R2021a includes GraphBLAS v3.3.3 as the
built-in method for C=A*B when A and B are sparse.

MATLAB (R2020a) was used for this demo, where the built-in
C=A*B relies on a single-threaded method, also by Tim Davis,
in SuiteSparse/MATLAB_Tools/SSMULT, introduced many years ago
into MATLAB.

