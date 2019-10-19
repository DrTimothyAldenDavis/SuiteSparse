function colamd_make
%COLAMD_MAKE compiles COLAMD2 and SYMAMD2 for MATLAB
%
% Example:
%   colamd_make
%
% See also colamd, symamd

% Copyright 2006, Timothy A. Davis, University of Florida


if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

mex -output colamd2mex -O -I../UFconfig colamdmex.c colamd.c colamd_global.c
mex -output symamd2mex -O -I../UFconfig symamdmex.c colamd.c colamd_global.c
fprintf ('COLAMD2 and SYMAMD2 successfully compiled.\n') ;
