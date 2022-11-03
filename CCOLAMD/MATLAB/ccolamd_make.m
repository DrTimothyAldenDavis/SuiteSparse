function ccolamd_make
%CCOLAMD_MAKE compiles CCOLAMD and CSYMAMD for MATLAB
%
% Example:
%   ccolamd_make
%
% See also ccolamd, csymamd

% CCOLAMD, Copyright (c) 2005-2022, Univ. of Florida, All Rights Reserved.
% Authors: Timothy A. Davis, Sivasankaran Rajamanickam, and Stefan Larimore.
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

src = '../Source/ccolamd_l.c ../../SuiteSparse_config/SuiteSparse_config.c' ;
cmd = sprintf ( ...
    'mex -O %s -I../../SuiteSparse_config -I../Include -output ', d) ;
s = [cmd 'ccolamd ccolamdmex.c ' src] ;

if (~(ispc || ismac))
    % for POSIX timing routine
    s = [s ' -lrt'] ;
end
if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;

s = [cmd 'csymamd csymamdmex.c ' src] ;

if (~(ispc || ismac))
    % for POSIX timing routine
    s = [s ' -lrt'] ;
end

if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;
fprintf ('CCOLAMD and CSYMAMD successfully compiled.\n') ;
