function amd_make
%AMD_MAKE to compile amd2 for use in MATLAB
%
% Example:
%   amd_make
%
% See also amd, amd2.

% AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
% Iain S. Duff.  All Rights Reserved.
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
cmd = sprintf ('mex -O %s -output amd2 %s amd_mex.c %s', d, i, ...
    '../../SuiteSparse_config/SuiteSparse_config.c') ;
files = {'amd_l_order', 'amd_l_dump', 'amd_l_postorder', 'amd_l_post_tree', ...
    'amd_l_aat', 'amd_l2', 'amd_l1', 'amd_l_defaults', 'amd_l_control', ...
    'amd_l_info', 'amd_l_valid', 'amd_l_preprocess' } ;
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

fprintf ('AMD successfully compiled.\n') ;
