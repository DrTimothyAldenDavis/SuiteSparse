function F = inverse (F)
%INVERSE "inverts" F by flagging it as the factorization of inv(A).
%
% Example
%
%   F = factorize (A) ;
%   S = inverse (F) ;
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

F.is_inverse = ~(F.is_inverse) ;

