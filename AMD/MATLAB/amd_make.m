function amd_make
%AMD_MAKE to compile amd2 for use in MATLAB
%
% Example:
%   amd_make
%
% See also amd, amd2.
%
% --------------------------------------------------------------------------
% Copyright 2006 by Timothy A. Davis,
% Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.
% email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.
% web: http://www.cise.ufl.edu/research/sparse/amd
% --------------------------------------------------------------------------

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

i = sprintf ('-I..%sInclude -I..%s..%sUFconfig', filesep, filesep, filesep) ;
cmd = sprintf ('mex -inline -O -output amd2 %s amd_mex.c', i) ;
files = {'amd_order', 'amd_dump', 'amd_postorder', 'amd_post_tree', ...
    'amd_aat', 'amd_2', 'amd_1', 'amd_defaults', 'amd_control', ...
    'amd_info', 'amd_valid', 'amd_global', 'amd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ..%sSource%s%s.c', cmd, filesep, filesep, files {i}) ;
end
% fprintf ('%s\n', cmd) ;
eval (cmd) ;
fprintf ('AMD successfully compiled.\n') ;
