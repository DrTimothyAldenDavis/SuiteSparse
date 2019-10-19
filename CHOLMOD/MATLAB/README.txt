-----------------------
Using CHOLMOD in MATLAB                         http://www.suitesparse.com
-----------------------

See Contents.m for a description of each CHOLMOD function.

To compile CHOLMOD for use in MATLAB, simply type cholmod_install at
the MATLAB command line, while in the CHOLMOD/MATLAB directory.

cholmod_demo is a short demo program for CHOLMOD.  Type "cholmod_demo" in
your MATLAB command window to test your newly compiling CHOLMOD functions.
Test/cholmod_test.m runs the test suite for the MATLAB interface to CHOLMOD.
It requires the "UFget" interface to the UF sparse matrix collection, but
provides a more extensive test for CHOLMOD.  To obtain a copy of UFget, see
http://www.suitesparse.com .

----------------------------------------
Using AMD, CCOLAMD, and COLAMD in MATLAB
----------------------------------------

The following steps are not required to use CHOLMOD in MATLAB.

To use AMD in MATLAB, go to the AMD/MATLAB directory and either type "amd_install"
in the MATLAB command window, or type "make" in the Unix shell.

To use CCOLAMD in MATLAB, go to the CCOLAMD directory and type ccolamd_install.

COLAMD is already an integral part of MATLAB, but you can upgrade to the most
recent version.  Go to the COLAMD directory and type "colamd_install" in

----------------------------------------
To install all of SuiteSparse for MATLAB
----------------------------------------

To install all these packages and more, simply go the the top level folder of
SuiteSparse.  You should see the SuiteSparse_install.m file there.  Type
SuiteSparse_install in the MATLAB command window.

After installing SuiteSparse, your path has been updated, but the changes
will not be saved for your next MATLAB session unless you save your path.
Type 'pathtool' and click 'save'.

