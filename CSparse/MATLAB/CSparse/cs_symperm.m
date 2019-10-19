function C = cs_symperm (A,p)                                               %#ok
%CS_SYMPERM symmetric permutation of a symmetric matrix.
%   C = cs_symperm(A,p) computes C = A(p,p), but accesses only the
%   upper triangular part of A, and returns C upper triangular (A and C are
%   symmetric with just their upper triangular parts stored).  A must be square.
%
%   Example:
%       Prob = UFget ('HB/bcsstk01') ; A = Prob.A ;
%       p = cs_amd (A) ;
%       C = cs_symperm (A, p) ;
%       cspy (A (p,p)) ;
%       cspy (C) ;
%       C - triu (A (p,p))
%
%   See also CS_PERMUTE, SUBSREF, TRIU.

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_symperm mexFunction not found') ;
