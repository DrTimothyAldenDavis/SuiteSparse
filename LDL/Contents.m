% Sparse LDL factorization
% 
%    ldlsparse - LDL' factorization of a real, sparse, symmetric matrix
%    ldldemo   - demo program for LDL
%    ldlrow    - an m-file description of the algorithm used by LDL
%    ldltest   - test program for LDL
%    ldlmain2  - compiles and runs a longer test program
%
% Example:
%
%	[L, D, Parent, fl] = ldlsparse (A)

% Copyright (c) 2005 by Timothy A. Davis.
% LDL Version 1.3

% LDL License:  GNU Lesser General Public License as published by the
%   Free Software Foundation; either version 2.1 of the License, or
%   (at your option) any later version.
%
% Availability:
%
%   http://www.cise.ufl.edu/research/sparse/ldl
%
% Acknowledgements:
%
%   This work was supported by the National Science Foundation, under
%   grant CCR-0203270.
%
%   Portions of this work were done while on sabbatical at Stanford University
%   and Lawrence Berkeley National Laboratory (with funding from the SciDAC
%   program).  I would like to thank Gene Golub, Esmond Ng, and Horst Simon
%   for making this sabbatical possible.
