function C = cs_permute (A,p,q)                                             %#ok
%CS_PERMUTE permute a sparse matrix.
%   C = cs_permute(A,p,q) computes C = A(p,q)
%
%   Example:
%       Prob = UFget ('HB/arc130') ; A = Prob.A ; [m n] = size (A) ;
%       p = randperm (m) ; q = randperm (n) ;
%       C = cs_permute (A,p,q) ;    % C = A(p,q)
%
%   See also CS_SYMPERM, SUBSREF.

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_permute mexFunction not found') ;
