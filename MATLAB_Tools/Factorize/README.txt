FACTORIZE:  an object-oriented method for solving linear systems and least
squares problems.

See the Factorize/Contents.m file for more details on how to install and test
this package.  For a demo, see Factorize/Demo (run "fdemo" from the MATLAB
Command window).  For additional documentation, see Factorize/Doc.

The COD function for sparse matrices requires the SPQR mexFunction from the
SuiteSparse library.  The simplest way to get this is to install all of
SuiteSparse from http://www.suitesparse.com.  The FACTORIZE method can be used
without SPQR; in this case, the COD for sparse matrices is not used.  This has
no effect on the use of this method for full-rank matrices, since COD is used
only for rank-deficient matrices.

Factorize, Copyright (c) 2011-2012, Timothy A Davis. All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause
