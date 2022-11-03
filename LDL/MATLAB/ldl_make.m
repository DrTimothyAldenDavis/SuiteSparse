function ldl_make
%LDL_MAKE compile LDL
%
% Example:
%   ldl_make       % compiles ldlsparse and ldlsymbol
%
% See also ldlsparse, ldlsymbol

% LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    d = ['-silent ' d] ;
end

eval (sprintf ('mex -O %s -I../../SuiteSparse_config -I../Include -output ldlsparse ../Source/ldll.c ldlmex.c', d)) ;
eval (sprintf ('mex -O %s -I../../SuiteSparse_config -I../Include -output ldlsymbol ../Source/ldll.c ldlsymbolmex.c', d)) ;
fprintf ('LDL successfully compiled.\n') ;

