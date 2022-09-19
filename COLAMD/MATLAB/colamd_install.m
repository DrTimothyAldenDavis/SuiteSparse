function colamd_install
%COLAMD_MAKE to compile and install the colamd2 and symamd2 mexFunction.
%   Your current directory must be COLAMD/MATLAB for this function to work.
%
% Example:
%   colamd_install
%
% See also colamd2, symamd2.

% COLAMD, Copyright (c) 1998-2022, Timothy A. Davis, and Stefan Larimore.
% SPDX-License-Identifier: BSD-3-clause

% Developed in collaboration with J. Gilbert and E. Ng.
% Acknowledgements: This work was supported by the National Science Foundation,
% under grants DMS-9504974 and DMS-9803599.

colamd_make
addpath (pwd)
fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n\n', pwd) ;
colamd_demo
