function C = cs_droptol (A, tol)                                            %#ok
%CS_DROPTOL remove small entries from a sparse matrix.
%   C = cs_droptol(A,tol) removes entries from A of magnitude less than or
%   equal to tol.  Same as A = A .* (abs (A) >= tol).
%
%   Example:
%       Prob = UFget ('HB/arc130') ; A = Prob.A ;
%       cspy (abs (A) >= 1e-10) ;
%       C = cs_droptol (A, 1e-10) ;
%       cspy (C) ;
%
%   See also: RELOP, ABS

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_droptol mexFunction not found') ;

