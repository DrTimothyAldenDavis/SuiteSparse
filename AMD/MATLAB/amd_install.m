function amd_install
%AMD_INSTALL compile and install amd2 for use in MATLAB
%   Your current directory must be AMD/MATLAB for this function to work.
%
% Example:
%   amd_install
%
% See also amd, amd2.

% AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
% Iain S. Duff.  All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

amd_make
addpath (pwd)
fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n\n', pwd) ;
amd_demo
