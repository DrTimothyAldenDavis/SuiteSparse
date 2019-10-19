function C = ssmult (A,B)						    %#ok
%SSMULT multiplies two sparse matrices.
% C = ssmult (A,B) computes C=A*B where A and B are sparse.  This function is
% typically faster than C=A*B in MATLAB 7.4, and always uses less memory.
% Either A or B, or both, can be complex.  Only matrices of class "double" are
% supported.
%
% Example:
%   load west0479
%   A = west0479 ;
%   B = sprand (west0479) ;
%   C = A*B ;
%   D = ssmult (A,B) ;
%   C-D
%
% See also ssmultsym, mtimes.

% Copyright 2007, Timothy A. Davis, University of Florida

error ('ssmult mexFunction not found') ;
