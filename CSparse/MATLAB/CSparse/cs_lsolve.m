function x = cs_lsolve (L,b)
%CS_LSOLVE solve a sparse lower triangular system L*x=b.
%   x = cs_lsolve(L,b) computes x = L\b, L must be lower triangular with a
%   zero-free diagonal.  b must be a column vector.  x is full if b is full.
%   If b is sparse, x is sparse but the nonzero pattern of x is NOT sorted (it
%   is returned in topological order).
%
%   See also CS_LTSOLVE, CS_USOLVE, CS_UTSOLVE, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_lsolve mexFunction not found') ;
