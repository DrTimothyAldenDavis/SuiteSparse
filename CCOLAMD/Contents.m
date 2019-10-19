% CCOLAMD, constrained approximate minimum degree ordering
%
% Primary functions:
%   csymamd        - constrained symmetric approximate minimum degree permutation
%   ccolamd        - constrained column approximate minimum degree permutation.
%
% helper and test functions:
%   ccolamd_demo   - demo for ccolamd and csymamd
%   ccolamd_make   - compiles ccolamd for MATLAB
%   ccolamd_test   - extensive test of ccolamd and csymamd
%   luflops        - compute the flop count for sparse LU factorization
%   ccolamdtestmex - test function for ccolamd
%   csymamdtestmex - test function for csymamd
%
% Example:
%   p = ccolamd (S, knobs, cmember)

% Copyright 2006, Timothy A. Davis
