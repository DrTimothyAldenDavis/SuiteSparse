% LINFACTOR
%
% This simple MATLAB function shows you how to use LU or CHOL to factor a matrix
% and then solve a linear system A*x=b.
%
% Files
%   linfactor - factorize a matrix, or use the factors to solve Ax=b.
%   lintests  - test linfactor with many different kinds of systems.
%   lintest   - test A*x=b, using linfactor, x=A\b, and (ack!) the explicit inv(A).
%
% Example:
%   F = linfactor (A) ;     % factor A, returning an object F
%   x = linfactor (F,b) ;   % solve Ax=b using F (the factorization of A)
%   lintests ;              % test linfactor with various kinds of systems

% Copyright 2007, Timothy A. Davis, University of Florida.
%
% License: this software is free for any use.  No warranty included or implied.
% You must agree to only one condition to use this software: you must be aware
% that you have been told that using inv(A) is a horrible, awful, and absoluty
% abysmal method for solving a linear system of equations.  If you do not
% agree to this condition, you must delete this software.
