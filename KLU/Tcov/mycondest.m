function c = mycondest(A)
%CONDEST 1-norm condition number estimate.
%   C = CONDEST(A) computes a lower bound C for the 1-norm condition
%   number of a square matrix A.
%
%   Example:
%	c = mycondest(A)
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
k = find(abs(diag(U))==0);						    %#ok
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
