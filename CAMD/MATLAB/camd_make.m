function camd_make
%CAMD_MAKE to compile camd for use in MATLAB
%
% Example:
%   camd_make
%
% See also camd.

% Copyright 1994-2007, Tim Davis, University of Florida,
% Patrick R. Amestoy, Iain S. Duff, and Yanqing Chen.

details = 0 ;	    % 1 if details of each command are to be printed

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

i = sprintf ('-I..%sInclude -I..%s..%sUFconfig', filesep, filesep, filesep) ;
cmd = sprintf ('mex -O %s -DDLONG -output camd %s camd_mex.c', d, i) ;
files = {'camd_order', 'camd_dump', 'camd_postorder', ...
    'camd_aat', 'camd_2', 'camd_1', 'camd_defaults', 'camd_control', ...
    'camd_info', 'camd_valid', 'camd_global', 'camd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ..%sSource%s%s.c', cmd, filesep, filesep, files {i}) ;
end
if (details)
    fprintf ('%s\n', cmd) ;
end
eval (cmd) ;

fprintf ('CAMD successfully compiled.\n') ;
