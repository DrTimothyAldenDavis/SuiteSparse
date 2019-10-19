function C = cs_symperm (A,p)
%CS_SYMPERM symmetric permutation of a symmetric matrix.
%   C = cs_symperm(A,p) computes C = A(p,p), but accesses only the
%   upper triangular part of A, and returns C upper triangular (A and C are
%   symmetric with just their upper triangular parts stored).  A must be square.
%
%   See also CS_PERMUTE, SUBSREF, TRIU.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_symperm mexFunction not found') ;
