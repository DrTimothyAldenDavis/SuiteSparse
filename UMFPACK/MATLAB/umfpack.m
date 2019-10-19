function [out1, out2, out3, out4, out5] = umfpack (in1, in2, in3, in4, in5)
% UMFPACK v5.0 is a MATLAB mexFunction for solving sparse linear systems.
%
% UMFPACK v5.0:                       |  MATLAB approximate equivalent:
% ---------------------------------------------------------------------
% x = umfpack (A, '\', b) ;           |  x = A \ b
%                                     |
% x = umfpack (b, '/', A) ;           |  x = b / A
%                                     |
% [L,U,P,Q] = umfpack (A) ;           |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  [L,U,P] = lu (A*Q) ;
%                                     |
% [L,U,P,Q,R] = umfpack (A) ;         |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  r = full (sum (abs (A), 2)) ;
%                                     |  r (find (r == 0)) = 1 ;
%                                     |  R = spdiags (r, 0, m, m) ;
%                                     |  [L,U,P] = lu ((R\A)*Q) ;
%                                     |
% [P,Q,F,C] = umfpack (A, 'symbolic') |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  [count,h,parent,post] = ...
%                                     |  symbfact (A*Q, 'col') ;
%
% A must be sparse.  It can be complex, singular, and/or rectangular.  A must be
% square for '/' or '\'.  b must be a full real or complex vector.  For
% [L,U,P,Q,R] = umfpack (A), the factorization is L*U = P*(R\A)*Q.  If A has a
% mostly symmetric nonzero pattern, then replace "colamd" with "amd" in the
% MATLAB-equivalent column in the table above.  Type umfpack_details for more
% information.
%
% See also: lu_normest, colamd, amd.
% To use UMFPACK for an arbitrary b, see umfpack_solve.

% UMFPACK Version 5.0, Copyright (c) 1995-2006 by Timothy A. Davis.
% All Rights Reserved.  Type umfpack_details for License.

help umfpack
error ('umfpack mexFunction not found!  Use umfpack_make to compile umfpack.') ;

