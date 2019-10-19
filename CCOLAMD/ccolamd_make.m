function ccolamd_make
%CCOLAMD_MAKE compiles ccolamd for MATLAB
% Example:
%   ccolamd_make
% See also ccolamd, csymamd

% Copyright 2006, Timothy A. Davis

mex -O -I../UFconfig -output ccolamd ccolamdmex.c ccolamd.c ccolamd_global.c
mex -O -I../UFconfig -output csymamd csymamdmex.c ccolamd.c ccolamd_global.c
fprintf ('CCOLAMD successfully compiled.\n') ;
