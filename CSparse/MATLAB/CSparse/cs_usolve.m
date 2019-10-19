function x = cs_usolve (U,b)
%CS_USOLVE solve a sparse upper triangular system U*x=b.
%   x = cs_usolve(U,b) computes x = U\b, U must be lower triangular with a
%   zero-free diagonal.  b must be a column vector.  x is full if b is full.
%   If b is sparse, x is sparse but nonzero pattern of x is NOT sorted (it is
%   returned in topological order).
%
%   See also CS_LSOLVE, CS_LTSOLVE, CS_UTSOLVE, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_usolve mexFunction not found') ;
