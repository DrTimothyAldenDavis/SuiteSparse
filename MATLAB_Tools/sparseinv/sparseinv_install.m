function sparseinv_install
%SPARSEINV_INSTALL compiles and installs the sparseinv function.
% Your current working directory must be the sparseinv directory for this
% function to work.
%
% Example:
%   sparseinv_install
%
% See also sparseinv, sparseinv_test

% SPARSEINV, Copyright (c) 2011, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

is64 = ~isempty (strfind (computer, '64')) ;
if (is64)
    fprintf ('Compiling sparseinv (64-bit)\n') ;
    mex -largeArrayDims sparseinv_mex.c sparseinv.c
else
    fprintf ('Compiling sparseinv (32-bit)\n') ;
    mex sparseinv_mex.c sparseinv.c
end
addpath (pwd)
