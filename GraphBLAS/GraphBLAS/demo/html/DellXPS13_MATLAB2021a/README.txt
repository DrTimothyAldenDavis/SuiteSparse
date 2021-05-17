Results on a Dell XPS 13 9380 laptop.

Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (up to 4.6GHz),
16GB of RAM.  MATLAB R2021a.  Ubuntu 18.04.  GraphBLAS and
the mexFunction interface compiled with gcc 7.5.0.  The CPU
has 4 hardware cores, and 4 threads were used (the default
when running OpenMP inside MATLAB).

Note that MATLAB R2021a includes GraphBLAS v3.3.3 as the
built-in method for C=A*B when A and B are sparse.

As a result, GraphBLAS is used for the built-in C=A*B for
MATLAB sparse matrices (using v3.3.3) and also for C=A*B
when A and/or B are @GrB matrices, using the current version
of GraphBLAS.

