RBio: Version 1.0, Dec 2, 2006.
A MATLAB Toolbox for reading/writing sparse matrices in Rutherford/Boeing
format.

NOTE: RBio is not yet ported to 64-bit MATLAB.

To install, using MATLAB:

    type "RBinstall" in the MATLAB command window.

To install, using the Makefile:

    type "make" at the Unix/Linux prompt.

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

  Copyright 2006, Timothy A. Davis

--------------------------------------------------------------------------------
Files and directories:
--------------------------------------------------------------------------------


    README.txt	    this file
    Contents.m	    MATLAB help for the RBio toolbox

    Test	    test matrices

    Makefile	    Makefile for compiling RBio (or use RBinstall in MATLAB)
    RBcread.f	    read a complex sparse matrix
    RBcsplit.f	    split a complex matrix into its real and imaginary parts
    RBfix.m	    read a possibly corrupted R/B file
    RBinstall.m	    compile RBio for use in MATLAB
    RBint.f	    determine the appropriate integer class
    RBraw.f	    read the raw contents of a R/B file
    RBraw.m	    MATLAB help for RBraw
    RBreade.m	    read a finite-element sparse matrix
    RBread.f	    utility routines for either real or complex matrices
    RBread.m	    MATLAB help for RBread_mex.f
    RBread_mex.f    mexFunction to read a real or complex sparse matrix
    RBrread.f	    read a real sparse matrix, compare with RBcread.f
    RBtype.f	    determine the Rutherford/Boeing type
    RBtype.m	    MATLAB help for RBtype
    RBwrite.f	    write a real or complex sparse matrix
    RBwrite.m	    MATLAB help for RBwrite_mex.f
    RBwrite_mex.f   mexFunction to write a real or complex sparse matrix

./Test:
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

