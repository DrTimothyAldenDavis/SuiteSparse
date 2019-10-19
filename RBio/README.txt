RBio: Copyright 2016, Timothy A. Davis, http://www.suitesparse.com
A MATLAB Toolbox for reading/writing sparse matrices in Rutherford/Boeing
format.

To install the MATLAB functions, cd to the RBio directory (the one containing
RBinstall.m, not the top-level directory) and type "RBinstall" in the MATLAB
command window.  To compile the C codes, type "make" (requires Unix).  For
examples on how to use the C interface, see Include/RBio.h and Demo/RBdemo.c.
To install the shared library into /usr/local/lib and /usr/local/include, do
"make install"; do "make uninstall" to remove the library.

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


--------------------------------------------------------------------------------
Files and directories:
--------------------------------------------------------------------------------

    README.txt	    this file
    Makefile        for Unix; 

./RBio: MATLAB interface

    Contents.m	    MATLAB help for the RBio toolbox
    RBfix.m	    read a possibly corrupted R/B file
    RBinstall.m	    compile and install RBio for use in MATLAB, and run tests
    RBmake.m	    compile RBio for use in MATLAB
    RBraw.m	    MATLAB help for RBraw
    RBreade.m	    read a finite-element sparse matrix
    RBread.m	    MATLAB help for RBread
    RBtype.m	    MATLAB help for RBtype
    RBwrite.m	    MATLAB help for RBwrite

    Makefile        for Unix; see also RBinstall.m and RBmake.m
    RBerror.c       error handling
    RBraw.c         RBraw mexFunction
    RBread.c        RBread mexFunction
    RBtype.c        RBtype mexFunction
    RBwrite.c       RBwrite mexFunction

./RBio/private:  test directory for MATLAB

    testRB1.m	    simple test for RBio
    testRB2.m	    simple test for RBio (requires UFget)
    testRB3.m       extensive test for RBio (requires UFget)

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
    License.txt	    license

./Demo: C demo program

    Makefile        for compiling the demo
    RBdemo.c        the demo itself

./Include: include files for user programs

    RBio.h

./Lib:

    Makefile        for compiling the RBio library

./Source: C source codes

    RBio.c          C-callable RBio functions

./Tcov: extensive test of C-callable RBio functions

    Makefile        for compiling the test
    RBtest.c        the test program
    README.txt      short help for Tcov

./Tcov/mangled: erroneous matrices for testing error-handling

    1.rb
    2.rb
    3.rb
    4.rb
    5.rb
    6.rb
    7.rb
    8.rb
    9.rb
    10.rb
    11.rb
    12.rb
    13.rb
    14.rb
    15.rb

./Tcov/matrices: test matrices

    Tina_DisCog.tar.gz
    dwg961a.tar.gz
    m4.rb
    m4b.rb
    mhd1280a.tar.gz
    mhd1280b.tar.gz
    plskz362.tar.gz
    qc324.tar.gz
    s4.rb
    west0067.tar.gz

