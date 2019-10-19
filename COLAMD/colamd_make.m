function colamd_make
% COLAMD_MAKE:  compiles COLAMD Version 2.5 for MATLAB

mex -O -I../UFconfig colamdmex.c colamd.c colamd_global.c
mex -O -I../UFconfig symamdmex.c colamd.c colamd_global.c
fprintf ('COLAMD successfully compiled.\n') ;
