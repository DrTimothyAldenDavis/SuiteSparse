-----------------------
Using CHOLMOD in MATLAB
-----------------------

See Contents.m for a description of each CHOLMOD function.

To compile CHOLMOD for use in MATLAB, you may optionally use METIS.  If you
do not use METIS, compile CHOLMOD with the -DNPARTITION flag.

There are two ways of compiling the CHOLMOD mexFunctions.  The 2nd one is best.

(1) Using the Unix "make" command.  This will compile the AMD, COLAMD, CCOLAMD,
    and CHOLMOD libraries (*.a).  You must first add -fexceptions to the
    CFLAGS definition in UFconfig/UFconfig.mk first (for Linux).  Otherwise,
    MATLAB will not be able to handle exceptions properly (CHOLMOD may terminate
    MATLAB if it encounters an error).  The METIS library must also be compiled
    with -fexceptions (see metis-4.0/Makefile.in).
    For other operating systems, see your default MATLAB mexopts.sh file
    (type "mex -v").  Next, simply type "make" in the operating system command
    window in this directory.

    On Linux (with gcc), you must compile all codes with the -fexceptions flag.
    See option (2) if you have problems compiling METIS.

(2) using the "cholmod_make" m-file in MATLAB.  First, place a copy of
    METIS 4.0.1 in the ../../metis-4.0 directory (the same directory that
    contains AMD, COLAMD, CCOLAMD, and CHOLMOD).
    Then, type "cholmod_make" in the MATLAB command window.
    All source files (including METIS) will be compiled with the MATLAB mex
    commnand.  This works on all operating systems, including Windows.
    You can use an alternate location for METIS, if you pass the pathname
    as the first argument to cholmod_make, as in

	cholmod_make ('path to your copy of metis-4.0 goes here') ;

    Option (2) is better because it allows for several workarounds for METIS.
    With Option (1), METIS terminates MATLAB if it runs out of memory.  That
    does not happen with Option (2).

You should also add CHOLMOD/MATLAB to your MATLAB path.

cholmod_demo is a short demo program for CHOLMOD.  Type "cholmod_demo" in
your MATLAB command window to test your newly compiling CHOLMOD functions.
Test/cholmod_test.m runs the test suite for the MATLAB interface to CHOLMOD.
It requires the "UFget" interface to the UF sparse matrix collection, but
provides a more extensive test for CHOLMOD.  To obtain a copy of UFget, see
http://www.cise.ufl.edu/research/sparse .

----------------------------------------
Using AMD, CCOLAMD, and COLAMD in MATLAB
----------------------------------------

The following steps are not required to use CHOLMOD in MATLAB.

To use AMD in MATLAB, go to the AMD/MATLAB directory and either type "amd_make"
in the MATLAB command window, or type "make" in the Unix shell.  Add AMD/MATLAB
to your MATLAB path.

To use CCOLAMD in MATLAB, go to the CCOLAMD directory and either type
"ccolamd_demo" in the MATLAB command window, or type "make" in the Unix shell.
Add CCOLAMD to your MATLAB path.

COLAMD is already an integral part of MATLAB, but you can upgrade to the most
recent version.  Go to the COLAMD directory and either type "colamd_demo" in
the MATLAB command window, or type "make" in the Unix shell.
Add COLAMD to your MATLAB path.

