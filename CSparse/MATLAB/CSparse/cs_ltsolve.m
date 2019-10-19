function x = cs_ltsolve (L,b)
%CS_LTSOLVE solve a sparse upper triangular system L'*x=b.
%   x = cs_ltsolve(L,b) computes x = L'\b, L must be lower triangular with a
%   zero-free diagonal.  b must be a full vector.
%
%   See also CS_LSOLVE, CS_USOLVE, CS_UTSOLVE, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_ltsolve mexFunction not found') ;
