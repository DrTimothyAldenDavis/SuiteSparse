function builtin_complex = GB_builtin_complex_get
%GB_BUILTINT_COMPLEX get the flag that determines the GrB_Type Complex
%
% builtin_complex = GB_builtin_complex_get
%
% Returns the boolean flag builtin_complex.  If true, GxB_FC64 is used,
% and set as the "user-defined" Complex type.  Otherwise, the Complex type is
% created as user-defined.
%
% See also GB_builtin_complex_set.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

global GraphBLAS_builtin_complex
if (isempty (GraphBLAS_builtin_complex))
    builtin_complex = GB_builtin_complex_set (true) ;
end
builtin_complex = GraphBLAS_builtin_complex ;

