function A = cs_frand (n,nel,s)                                             %#ok
%CS_FRAND generate a random finite-element matrix
% A = cs_frand (n,nel,s) creates an n-by-n sparse matrix consisting of nel
% finite elements, each of which are of size s-by-s with random symmetric
% nonzero pattern, plus the identity matrix.
%
% Example
%   A = cs_frand (100, 100, 3) ;
% See also cs_demo.

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_frand mexFunction not found') ;

