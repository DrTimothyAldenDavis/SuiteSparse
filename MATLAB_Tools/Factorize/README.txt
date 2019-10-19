FACTORIZE:  an object-oriented method for solving linear systems and least
squares problems.

See the Factorize/Contents.m file for more details on how to install and test
this package.  For a demo, see Factorize/Demo (run "fdemo" from the MATLAB
Command window).  For additional documentation, see Factorize/Doc.

The cod function for sparse matrices requires the SPQR mexFunction from the
SuiteSparse library.  The simplest way to get this is to install all of
SuiteSparse from http://www.cise.ufl.edu/research/sparse .  The factorize
method can be used without SPQR; in this case, the COD for sparse matrices
is not used.  This has no effect on the use of this method for full-rank
matrices, since COD is used only for rank-deficient matrices.

Copyright 2011, Timothy A. Davis, University of Florida.
