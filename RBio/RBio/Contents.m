% RBio: MATLAB toolbox for reading/writing sparse matrices in the Rutherford/
%   Boeing format, and for reading/writing problems in the SuiteSparse Matrix
%   Collection from/to a set of files in a directory.
%
%   RBread    - read a sparse matrix from a Rutherford/Boeing file
%   RBreade   - read a symmetric finite-element matrix from a R/B file
%   RBtype    - determine the Rutherford/Boeing type of a sparse matrix
%   RBwrite   - write a sparse matrix to a Rutherford/Boeing file
%   RBraw     - read the raw contents of a Rutherford/Boeing file
%   RBfix     - read a possibly corrupted matrix from a R/B file
%   RBinstall - install the RBio toolbox for use in MATLAB
%   RBmake    - compile the RBio toolbox for use in MATLAB
%
% Example:
%
%   load west0479
%   C = west0479 ;
%   RBwrite ('mywest', C, 'WEST0479 chemical eng. problem', 'west0479')
%   A = RBread ('mywest') ;
%   norm (A-C,1)
%
% See also ssget, mread, mwrite.

% RBio, Copyright (c) 2009-2022, Timothy A. Davis.  All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+
