WAITMEX provides a simple way of using a waitbar from within a C mexFunction.

Files:

    README.txt	    this file
    waitexample.c   a C mexFunction that uses the waitmex.c functions
    waitexample.m   help for waitexample
    waitex.m	    a MATLAB m-file equivalent of waitexample.m
    waitmex.c	    four functions for accessing a waitbar in a mexFunction
    waitmex.h	    include file required for using waitmex.c
    waitmex.m	    compiles the waitexample mexFunction

For more help, and to compile, install, and test the waitmex functions, type:

    waitmex

in the MATLAB Command Window.  For details on how to call the waitmex functions,
see waitmex.c, and see the examples given in waitexample.c.

These functions should be easily adaptable to any of the many replacements for
waitbar posted on the MATLAB Central File Exchange, particularly if they use
the same input and output arguments as the MATLAB waitbar.

Copyright 2007, Timothy A. Davis, http://www.suitesparse.com
