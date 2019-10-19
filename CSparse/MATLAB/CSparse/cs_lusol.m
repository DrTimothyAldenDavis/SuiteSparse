function x = cs_lusol (A,b,order,tol)
%CS_LUSOL solve Ax=b using LU factorization.
%   x = cs_lusol(A,b) computes x = A\b, where A is sparse and square, and b is a
%   full vector.  The ordering cs_amd(A,2) is used.
%
%   x = cs_lusol(A,b,1) also computes x = A\b, but uses the cs_amd(A) ordering
%   with diagonal preference (tol=0.001).
%
%   x = cs_lusol(A,b,order,tol) allows both the ordering and tolerance to be
%   defined.  The ordering defaults to 1, and tol defaults to 1.
%   ordering: 0: natural, 1: amd(A+A'), 2: amd(S'*S) where S=A except with no
%   dense rows, 3: amd(A'*A).
%
%   See also CS_LU, CS_AMD, CS_CHOLSOL, CS_QRSOL, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_lusol mexFunction not found') ;
