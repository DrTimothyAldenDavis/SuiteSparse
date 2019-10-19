function C = sdmult (S,F,transpose)
%SDMULT: sparse matrix times dense matrix
%   Compute C = S*F or S'*F where S is sparse and F is full (C is also sparse).
%   S and F must both be real or both be complex.  This function is
%   substantially faster than the MATLAB expression C=S*F when F has many
%   columns.
%
%   Usage:
%
%	C = sdmult (S,F) ;		C = S*F
%	C = sdmult (S,F,0) ;		C = S*F
%	C = sdmult (S,F,1) ;		C = S'*F
%
%   See also MTIMES

%   Copyright 2006, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

error ('sdmult mexFunction not found') ;


