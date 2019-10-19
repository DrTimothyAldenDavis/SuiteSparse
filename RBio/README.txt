RBio: Version 1.1.2, May 1, 2009.  A MATLAB Toolbox for reading/writing sparse
matrices in Rutherford/Boeing format.

To install, cd to the RBio directory and type "RBinstall" in the MATLAB
command window.  RBio is written in Fortran because the Rutherford/Boeing
format can require Fortran I/O statements, depending on how the files are
stored.  Files created by RBio do not require the Fortran I/O library to read
them, however.

--------------------------------------------------------------------------------
MATLAB help for RBio:
--------------------------------------------------------------------------------

  RBio - MATLAB toolbox for reading/writing sparse matrices in the Rutherford/
    Boeing format, and for reading/writing problems in the UF Sparse Matrix
    Collection from/to a set of files in a directory.

    RBread    - read a sparse matrix from a Rutherford/Boeing file
    RBreade   - read a symmetric finite-element matrix from a R/B file
    RBtype    - determine the Rutherford/Boeing type of a sparse matrix
    RBwrite   - write a sparse matrix to a Rutherford/Boeing file
    RBraw     - read the raw contents of a Rutherford/Boeing file
    RBfix     - read a possibly corrupted matrix from a R/B file
    RBinstall - install the RBio toolbox for use in MATLAB

  Example:

    load west0479
    C = west0479 ;
    RBwrite ('mywest', C, 'WEST0479 chemical eng. problem', 'west0479')
    A = RBread ('mywest') ;
    norm (A-C,1)

  See also UFget, mread, mwrite.

  Copyright 2007, Timothy A. Davis

--------------------------------------------------------------------------------
Files and directories:
--------------------------------------------------------------------------------

    README.txt	    this file
    Contents.m	    MATLAB help for the RBio toolbox

    RBfix.m	    read a possibly corrupted R/B file
    RBinstall.m	    compile and install RBio for use in MATLAB, and run tests
    RBmake.m	    compile RBio for use in MATLAB
    RBraw.m	    MATLAB help for RBraw
    RBreade.m	    read a finite-element sparse matrix
    RBread.m	    MATLAB help for RBread
    RBtype.m	    MATLAB help for RBtype
    RBwrite.m	    MATLAB help for RBwrite

    RBcread_32.f    read a complex sparse matrix, compare with RBrread_*.f
    RBcread_64.f

    RBcsplit_32.f   split a complex matrix into its real and imaginary parts
    RBcsplit_64.f

    RBraw_mex_32.f  mexFunction to read the raw contents of a R/B file
    RBraw_mex_64.f

    RBread_32.f	    utility routines for either real or complex matrices
    RBread_64.f

    RBread_mex_32.f mexFunction to read a real or complex sparse matrix
    RBread_mex_64.f

    RBrread_32.f    read a real sparse matrix, compare with RBcread_*.f
    RBrread_64.f

    RBtype_mex_32.f mexFunction to determine the Rutherford/Boeing type
    RBtype_mex_64.f

    RBwrite_32.f    write a real or complex sparse matrix
    RBwrite_64.f

    RBwrite_mex_32.f mexFunction to write a real or complex sparse matrix
    RBwrite_mex_64.f

./Test: directory with test codes and matrices

    testRB1.m	    simple test script for RBio
    testRB2.m	    simple test script for RBio (requires UFget)
    bcsstk01.rb	    HB/bcsstk01 Problem.A from UF Sparse Matrix Collection
    farm.rb	    Meszaros/farm Problem.A from UF Sparse Matrix Collection
    lap_25.pse	    original Harwell/Boeing version of lap_25 (finite-element)
    lap_25.rb	    HB/lap_25 Problem.A from UF Sparse Matrix Collection
    west0479.rb	    sparse matrix west0479 from UF Sparse Matrix Collection
    west0479.rua    original Harwell/Boeing version of west0479

    Note that the west0479 matrix provided in the Test directory is the correct
    version.  The MATLAB statement "load west0479" gives you a matrix that is
    slightly incorrect (as of MATLAB Version 7.3, R2006b).

./Doc: directory with additional documentation and license

    ChangeLog	    changes since first release
    dodiff	    compare 32-bit and 64-bit codes, real and complex
    gpl.txt	    GNU license
    License.txt	    license

