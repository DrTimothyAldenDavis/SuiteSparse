function amd_make
%AMD_MAKE to compile amd2 for use in MATLAB
%
% Example:
%   amd_make
%
% See also amd, amd2.

% Copyright 1994-2007, Tim Davis, University of Florida,
% Patrick R. Amestoy, and Iain S. Duff. 

details = 0 ;	    % 1 if details of each command are to be printed

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

i = sprintf ('-I..%sInclude -I..%s..%sUFconfig', filesep, filesep, filesep) ;
cmd = sprintf ('mex -O %s -DDLONG -output amd2 %s amd_mex.c', d, i) ;
files = {'amd_order', 'amd_dump', 'amd_postorder', 'amd_post_tree', ...
    'amd_aat', 'amd_2', 'amd_1', 'amd_defaults', 'amd_control', ...
    'amd_info', 'amd_valid', 'amd_global', 'amd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ..%sSource%s%s.c', cmd, filesep, filesep, files {i}) ;
end
if (details)
    fprintf ('%s\n', cmd) ;
end
eval (cmd) ;

fprintf ('AMD successfully compiled.\n') ;
