function x = cs_qrsol (A,b,order)
%CS_QRSOL solve a sparse least-squares problem.
%   x = cs_qrsol(A,b) solves the over-determined least squares problem to
%   find x that minimizes norm(A*x-b), where b is a full vector and
%   A is m-by-n with m >= n.  If m < n, it solves the underdetermined system
%   Ax=b.  A 3rd input argument specifies the ordering method to use
%   (0: natural, 3: amd(A'*A)).
%
%   See also CS_QR, CS_AMD, CS_LUSOL, CS_CHOLSOL, MLDIVIDE.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_qrsol mexFunction not found') ;
