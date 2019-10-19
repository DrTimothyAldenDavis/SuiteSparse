function ldl_make
%LDL_MAKE compile LDL
%
% Example:
%   ldl_make       % compiles ldlsparse and ldlsymbol
%
% See also ldlsparse, ldlsymbol

% Copyright 2006-2007 by Timothy A. Davis, Univ. of Florida

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end
eval (sprintf ('mex -O %s -DLDL_LONG -I../../UFconfig -I../Include -output ldlsparse ../Source/ldl.c ldlmex.c', d)) ;
eval (sprintf ('mex -O %s -DLDL_LONG -I../../UFconfig -I../Include -output ldlsymbol ../Source/ldl.c ldlsymbolmex.c', d)) ;
fprintf ('LDL successfully compiled.\n') ;

