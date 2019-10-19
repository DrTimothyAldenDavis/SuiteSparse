function F = minus (F,w)
%MINUS downdate a dense Cholesky factorization
%
% Example
%   % F becomes the Cholesky factorization of A-w*w'
%   F = factorize (A) ;
%   w = rand (size (A,1),1) ;
%   F = F - w ;
%   x = F\b ;               % computes x = (A-w*w')\b
%
% See also factorize, cholupdate.

% Copyright 2009, Timothy A. Davis, University of Florida

if (F.kind ~= 6)
    error ('Only dense Cholesky factorization downdate supported.') ;
end

F.R = cholupdate (F.R, w, '-') ;
F.A = F.A - w*w' ;

