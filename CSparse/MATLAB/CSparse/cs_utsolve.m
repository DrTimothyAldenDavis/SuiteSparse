function x = cs_utsolve (U,b)
%CS_UTSOLVE solve a sparse lower triangular system U'*x=b.
%   x = cs_utsolve(U,b) computes x = U'\b, U must be upper triangular with a
%   zero-free diagonal.  b must be a full vector.
%
%   See also CS_LSOLVE, CS_LTSOLVE, CS_USOLVE, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_utsolve mexFunction not found') ;
