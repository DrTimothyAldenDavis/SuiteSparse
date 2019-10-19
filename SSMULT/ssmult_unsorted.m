function C = ssmult_unsorted (A,B)					    %#ok
%SSMULT_UNSORTED multiplies two sparse matrices, returning non-standard result.
% C = ssmult_unsorted (A,B) computes C=A*B where A and B are sparse.  It returns
% C with unsorted row indices in its columns, and possibly with explicit zero
% entries due to numeric cancellation.  This does *NOT* conform to the MATLAB
% standard (as of MATLAB 7.4 ...) for MATLAB sparse matrices.  However, such
% matrices are often safe to use in subsequent operations (such as C*X where 
% X is a full matrix).   Computing C'' (a double transpose) gives a sorted
% result.  This function is typically MUCH faster than C=A*B in MATLAB 7.4, and
% uses less memory.  Either A or B, or both, can be complex.  Only matrices of
% class "double" are supported.  The primary reason for this function is to
% demonstrate how much performance MATLAB loses by insisting on keeping its
% sparse matrices sorted.
%
% *** USE AT YOUR OWN RISK.  USE SSMULT TO BE SAFE. ***
%
% Example:
%   load west0479
%   A = west0479 ;
%   B = sprand (west0479) ;
%   tic ; C = A*B ; toc
%   tic ; D = ssmult_unsorted (A,B) ; toc
%   C-D
%
% To see that the result D from ssmult_unsorted has unsorted columns:
%
%   spparms ('spumoni', 1)
%   colamd (D) ;
%   spparms ('spumoni', 0)
%
% See also ssmult, ssmultsym, mtimes.
%
% *** USE AT YOUR OWN RISK.  USE SSMULT TO BE SAFE. ***

% Copyright 2007, Timothy A. Davis, University of Florida

error ('ssmult_unsorted mexFunction not found') ;
