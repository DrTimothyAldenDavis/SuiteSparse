function S = inverse (X)
%INVERSE factorized representation of inv(A).
% INVERSE is a fast and accurate replacement for INV when you want to solve
% a linear system or least squares problem, or when you want to multiply
% something by the inverse of A.  The inverse itself is NOT computed,
% UNLESS the factorized form of the inverse is converted into a matrix via
% double(inverse(A)).  If A is rectangular and has full rank, inverse(A) is
% a factorized form of the pseudo-inverse of A.
%
% Example
%
%   x = inv(A)*b ;      % slow and inaccurate way to solve A*x=b
%   x = inverse(A)*b ;  % fast an accurate way to solve A*x=b (uses x=A\b)
%   x = A\b ;           % same as inverse(A)*b
%
%   x1 = A\b1 ; x2 = A\b2 ;                        % accurate but slow
%   S = inverse(A) ; x1 = S*b1 ; x2 = S*b1 ;       % fast and accurate
%   S = inv(A)     ; x1 = S*b1 ; x2 = S*b1 ;       % slow and inaccurate
%
%   Z = double(inverse(A)) ; % same as Z=inv(A), computes the inverse of
%                            % A, returning Z as a matrix, not an object.
%
%   F = factorize(A) ;  % computes the factorization of A
%   S = inverse(F) ;    % no flops, flags S as a factorized form of inv(A)
%
% Never use inv to multiply the inverse of a matrix A by another matrix.
% There are rare uses for the explicit inv(A), but never do inv(A)*B or
% B*inv(A).  Never do Z=A\eye(n), which is just the same thing as Z=inv(A).
%
% "Don't let that inv go past your eyes; to solve that system, factorize!"
%
% See also factorize, factorize1, mldivide, pinv
% Do not see inv!

% Copyright 2009, Timothy A. Davis, University of Florida

% This function is only called when X is a matrix.  If X is a factorize
% object, then factorize.inverse is called instead.

S = inverse (factorize (X)) ;
