% BTF ordering toolbox:
%
% Primary functions:
%
%   btf        - permute a square sparse matrix into upper block triangular form
%   maxtrans   - permute the columns of a sparse matrix so it has a zero-free diagonal
%   strongcomp - symmetric permutation to upper block triangular form
%
% Other:
%   btf_install - compile and install BTF for use in MATLAB.
%   btf_demo    - demo for BTF
%   drawbtf     - plot the BTF form of a matrix
%   btf_make    - compile BTF for use in MATLAB
%
% Example:
%   q = maxtrans (A)
%   [p,q,r] = btf (A)
%   [p,r] = strongcomp (A)

% BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Author: Timothy A. Davis.
% SPDX-License-Identifier: LGPL-2.1+

