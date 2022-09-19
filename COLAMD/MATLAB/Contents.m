% COLAMD, column approximate minimum degree ordering
%
% Primary:
%   colamd2     - Column approximate minimum degree permutation.
%   symamd2     - SYMAMD Symmetric approximate minimum degree permutation.
%
% helper and test functions:
%   colamd_demo - demo for colamd, column approx minimum degree ordering algorithm
%   colamd_make - compiles COLAMD2 and SYMAMD2 for MATLAB
%   colamd_make - compiles and installs COLAMD2 and SYMAMD2 for MATLAB
%   colamd_test - test colamd2 and symamd2
%   luflops     - compute the flop count for sparse LU factorization
%
% Example:
%   p = colamd2 (A)
%

% COLAMD, Copyright (c) 1998-2022, Timothy A. Davis, and Stefan Larimore.
% SPDX-License-Identifier: BSD-3-clause

% Developed in collaboration with J. Gilbert and E. Ng.
% Acknowledgements: This work was supported by the National Science Foundation,
% under grants DMS-9504974 and DMS-9803599.
