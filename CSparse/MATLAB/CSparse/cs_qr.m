function [V,beta,p,R,q] = cs_qr (A)
%CS_QR sparse QR factorization.
%   [V,beta,p,R] = cs_qr(A) computes the QR factorization of A(p,:).
%   [V,beta,p,R,q] = cs_qr(A) computes the QR factorization of A(p,q).
%   The fill-reducing ordering q is found via q = cs_amd(A,3).
%   The orthogonal factor Q can be obtained via
%   Q = cs_qright(V,beta,p,peye(size(V,1))), in which case Q*R=A(:,q) is the
%   resulting factorization.  A must be m-by-n with m >= n.  If A is
%   structurally rank deficient, additional empty rows may have been added to
%   V and R.
%
%   See also CS_AMD, CS_QRIGHT, CS_QR, CS_DMPERM, QR, COLAMD.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_qr mexFunction not found') ;
