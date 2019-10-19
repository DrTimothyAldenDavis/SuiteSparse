function S = double (F)
%DOUBLE returns the factorization as a single matrix, A or inv(A)
%
% Example
%   F = factorize (A) ;         % factorizes A
%   C = double(F) ;             % C = A
%   S = inv (A)                 % two ways to compute the explicit inverse
%   S = double (inverse (A))
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

% let factorize.subsref do all the work
ij.type = '()' ;
ij.subs = {':',':'} ;
S = subsref (F, ij) ;

