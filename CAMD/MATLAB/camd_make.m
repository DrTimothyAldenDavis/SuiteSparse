function camd_make
%CAMD_MAKE to compile camd for use in MATLAB
%
% Example:
%   camd_make
%
% See also camd.

% CAMD, Copyright (c) 2007-2022, Timothy A. Davis, Yanqing Chen, Patrick R.
% Amestoy, and Iain S. Duff.  All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

details = 0 ;	    % 1 if details of each command are to be printed

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    d = ['-silent ' d] ;
end

i = sprintf ('-I../Include -I../../SuiteSparse_config') ;
cmd = sprintf ('mex -O %s -output camd %s camd_mex.c %s', d, i, ...
    '../../SuiteSparse_config/SuiteSparse_config.c') ;
files = {'camd_l_order', 'camd_l_dump', 'camd_l_postorder', ...
    'camd_l_aat', 'camd_l2', 'camd_l1', 'camd_l_defaults', 'camd_l_control', ...
    'camd_l_info', 'camd_l_valid', 'camd_l_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ../Source/%s.c', cmd, files {i}) ;
end
if (details)
    fprintf ('%s\n', cmd) ;
end

if (~(ispc || ismac))
    % for POSIX timing routine
    cmd = [cmd ' -lrt'] ;
end

eval (cmd) ;

fprintf ('CAMD successfully compiled.\n') ;
