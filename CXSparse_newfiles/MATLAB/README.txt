CXSparse/MATLAB directory, which contains the MATLAB mexFunction interfaces
for CXSparse, demos, and tests.  It includes various "textbook" files
that are printed in the book, but not a proper part of CSparse itself.
It also includes "UFget", a MATLAB interface for the UF Sparse Matrix
Collection.

Type the command "cs_install" while in this directory.  It will compile
CSparse, and add the directories:

    CXSparse/MATLAB/CSparse
    CXSparse/MATLAB/Demo
    CXSparse/MATLAB/UFget

to your MATLAB path (see the "pathtool" command to add these to your path
permanently, for future MATLAB sessions).

To run the MATLAB demo programs, run cs_demo in the Demo directory.
This demo will work whether or not your compiler supports the complex type.

To run the MATLAB test programs, run testall in the Test directory.
However, you may run the tests in the Test directory only if your compiler
supports the ANSI C99 complex type.  If it does not support the ANSI C99
complex type, the tests in the Test directory will fail, since the codes there
test both the real and complex cases in CXSparse.
