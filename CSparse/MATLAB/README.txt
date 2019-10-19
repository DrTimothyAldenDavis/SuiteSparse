CSparse/MATLAB directory, which contains the MATLAB mexFunction interfaces
for CSparse, demos, and tests.  It includes various "textbook" files
that are printed in the book, but not a proper part of CSparse itself.
It also includes "UFget", a MATLAB interface for the UF Sparse Matrix
Collection.

Type the command "cs_install" while in this directory.  It will compile
CSparse, and add the directories:

    CSparse/MATLAB/CSparse
    CSparse/MATLAB/Demo
    CSparse/MATLAB/UFget

to your MATLAB path (see the "pathtool" command to add these to your path
permanently, for future MATLAB sessions).  It will also add the path

    CSparse/MATLAB/UFget

to your java class path (see the "javaaddpath" command).  Edit your
classpath.txt file (type the command "which claspath.txt") to add this
directory to your Java class path permanently.

To run the MATLAB demo programs, run cs_demo in the Demo directory.
To run the MATLAB test programs, run testall in the Test directory.
