function camd_make
%CAMD_MAKE to compile camd for use in MATLAB
%
% Example:
%   camd_make
%
% See also camd.
%
% --------------------------------------------------------------------------
% Copyright 2006 by Timothy A. Davis, Yanqing Chen,
% Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.
% email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.
% web: http://www.cise.ufl.edu/research/sparse/camd
% --------------------------------------------------------------------------

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

i = sprintf ('-I..%sInclude -I..%s..%sUFconfig', filesep, filesep, filesep) ;
cmd = sprintf ('mex -inline -O -output camd %s camd_mex.c', i) ;
files = {'camd_order', 'camd_dump', 'camd_postorder', ...
    'camd_aat', 'camd_2', 'camd_1', 'camd_defaults', 'camd_control', ...
    'camd_info', 'camd_valid', 'camd_global', 'camd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ..%sSource%s%s.c', cmd, filesep, filesep, files {i}) ;
end
% fprintf ('%s\n', cmd) ;
eval (cmd) ;
fprintf ('CAMD successfully compiled.\n') ;
