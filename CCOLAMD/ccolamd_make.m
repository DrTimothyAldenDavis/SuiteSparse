function ccolamd_make
% CCOLAMD_MAKE:  compiles CCOLAMD Version 2.5 for MATLAB

mex -O -I../UFconfig -output ccolamd ccolamdmex.c ccolamd.c ccolamd_global.c
mex -O -I../UFconfig -output csymamd csymamdmex.c ccolamd.c ccolamd_global.c
fprintf ('CCOLAMD successfully compiled.\n') ;
