function c = mycondest(A)
%function c = mycondest(A, t)
%CONDEST 1-norm condition number estimate.
%   C = CONDEST(A) computes a lower bound C for the 1-norm condition
%   number of a square matrix A.
%
%   C = CONDEST(A,T) changes T, a positive integer parameter equal to
%   the number of columns in an underlying iteration matrix.  Increasing the
%   number of columns usually gives a better condition estimate but increases
%   the cost.  The default is T = 2, which almost always gives an estimate
%   correct to within a factor 2.
%
%   [C,V] = CONDEST(A) also computes a vector V which is an approximate null
%   vector if C is large.  V satisfies NORM(A*V,1) = NORM(A,1)*NORM(V,1)/C.
%
%   Note: CONDEST invokes RAND.  If repeatable results are required then
%   invoke RAND('STATE',J), for some J, before calling this function.
%
%   Uses block 1-norm power method of Higham and Tisseur.
%
%   See also NORMEST1, COND, NORM.

%   Nicholas J. Higham, 9-8-99
%   Copyright 1984-2003 The MathWorks, Inc. 
%   $Revision: 5.18.4.1 $  $Date: 2003/05/01 20:42:31 $

if size(A,1) ~= size(A,2)
   error('MATLAB:condest:NonSquareMatrix', 'Matrix must be square.')
end
if isempty(A), c = 0;
    return, 
end

[L,U] = lu(A);
%if U has zero on diagonal, condition number
%is inf
k = find(abs(diag(U))==0);
if ~isempty(k)
   c = Inf;
else
%   if t == 1
    if isreal(A)
	Ainv_norm = mynormest1(L,U);
    else
	Ainv_norm = complex_norm(L, U) ;
%   elseif t == 2
%	Ainv_norm = mynormest(A,L,U);
   end
   A_norm = norm(A,1);
%   fprintf ('mycondest : A norm %g  Ainv norm %g\n',A_norm, Ainv_norm) ; 
   c = Ainv_norm*A_norm;
end
