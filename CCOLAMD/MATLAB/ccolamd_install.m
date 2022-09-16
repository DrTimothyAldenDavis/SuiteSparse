function ccolamd_install
%CCOLAMD_INSTALL compiles and installs ccolamd and csymamd for MATLAB
%   Your current directory must be CCOLAMD/MATLAB for this function to work.
%
% Example:
%   ccolamd_install
%
% See also ccolamd, csymamd.

% CCOLAMD, Copyright (c) 2005-2022, Univ. of Florida, All Rights Reserved.
% Authors: Timothy A. Davis, Sivasankaran Rajamanickam, and Stefan Larimore.
% SPDX-License-Identifier: BSD-3-clause

ccolamd_make
addpath (pwd)
fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n\n', pwd) ;
ccolamd_demo
