function btf_install
%BTF_INSTALL compile and install BTF for use in MATLAB.
% Your current working directory must be BTF/MATLAB for this function to work.
%
% Example:
%   btf_install
%
% See also btf, maxtrans, stroncomp, dmperm.

% BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Author: Timothy A. Davis.
% SPDX-License-Identifier: LGPL-2.1+

btf_make
addpath (pwd) ;
fprintf ('BTF has been compiled and installed.  The path:\n') ;
disp (pwd) ;
fprintf ('has been added to your path.  Use pathtool to add it permanently.\n');
btf_demo
