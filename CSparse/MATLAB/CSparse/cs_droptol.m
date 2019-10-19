function C = cs_droptol (A, tol)
%CS_DROPTOL remove small entries from a sparse matrix.
%   C = cs_droptol(A,tol) removes entries from A of magnitude less than or
%   equal to tol.  Same as A = A .* (abs (A) >= tol).

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_droptol mexFunction not found') ;

