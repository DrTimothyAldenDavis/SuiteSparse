function b = rhs (m)
% b = rhs (m), compute a right-hand-side
% Example:
%   b = rhs (30) ;
% See also: cs_demo

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

b = ones (m,1) + (0:m-1)'/m ;
