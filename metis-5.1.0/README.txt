METIS 5.1.0, with minor modifications by Tim Davis
to incorporate it into SuiteSparse.

SuiteSparse can work with the unmodified version
of METIS 5.1.0.  These changes are optional, unless
you want to use METIS in MATLAB.

(1) In metis-5.1.0/include/metis.h, the default integer size has been changed
    from 32 bits to 64 (IDXTYPEWIDTH).  METIS 5.1.0 gives this flexility to the
    user, asking the user to modify this file.  That has been done here.  You
    may instead use the original METIS 5.1.0, with 32-bit integers.  When
    compiled, SuiteSparse will properly detect the size of the METIS idx_t, and
    it will work gracefully with both versions.  The only constraint is that
    SuiteSparse will not be able to use METIS on the largest problems, if you
    limit its version of METIS to 32-bit integers.

(2) The files b64.c, rw.c, seq.c, timers.c, in metis-5.1.0/GKlib, and the files
    coarsen.c, fm.c, macros.h, mcutil.c, mincon.c, ometis.c, in
    metis-5.1.0/libmetis had C++ style comments (//) which break some C
    compilers (the MATLAB mex command on Linux, in particular).  They have been
    removed.  If your compiler is OK with //-style comments, then this fix
    is optional.

(3) The files metis-5.1.0/GKlib/GKLib.h and metis-5.1.0/GKlib/memory.c have
    been modified to disable the signal-handing in METIS when used via the
    MATLAB interface to METIS in CHOLMOD/MATLAB.  These signals are used when
    METIS runs out of memory, but they break MATLAB (you will get a segfault).
    This change is essential if METIS is to be used in MATLAB.

(4) The abs and iabs functions in the original metis-5.1.0/libmetis/parmetis.c
    and metis-5.1.0/libmetis/balance.c give compiler warnings when IDXTYPEWIDTH
    is 64, so they have been replaced with a type-agnostic macro, ABS.  This is
    just a compiler warning, so the fix is optional.

See http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
for the primary distrubtion of METIS, by George Karypis
University of Minnesota.

Tim Davis, Feb 1, 2016, Texas A&M University

