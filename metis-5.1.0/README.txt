METIS 5.1.0, with minor modifications by Tim Davis
to incorporate it into SuiteSparse.

SuiteSparse can work with the unmodified version
of METIS 5.1.0.  These changes are optional, unless
you want to use METIS in MATLAB.

(1) The integer type, idx_t, has been set to 64 bits
in metis-5.1.0 include/metis.h.  This is the primary change.
If you compile SuiteSparse with the original METIS, it will
work but SuiteSparse will be unable to use METIS on the very
largest matrices.

(2) with this change to 64 bit integers, a compiler warning
is generated in two files, regarding the use of abs and iabs.
These two files (balance.c and parmetis.c) have been modified
to avoid that warning.

(3) modified the memory manager to use the MATLAB malloc,
calloc, realloc, and free equivalents when compiling METIS
for MATLAB (GKlib/GKlib.h).  Disabled the internal signal
handling of METIS.  Has no effect on the compiled
libmetis.so (libmetis.dylib on the Mac).  This has no
effect on programs outside of MATLAB, but only when METIS
is used inside a MATLAB mexFunction.

(4) modified many *.c and *.h files to remove C++ comments
of the form "//".  These cause some C compilers to break
(in particular, the C compiler flags used by the MATLAB
mex command on Linux).

See http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
for the primary distrubtion of METIS, by George Karypis
University of Minnesota.

Tim Davis, Jan 30, 2016, Texas A&M University

