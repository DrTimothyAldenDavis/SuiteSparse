function C = GB_spec_apply (C, Mask, accum, op, A, descriptor)
%GB_SPEC_APPLY a MATLAB mimic of GrB_apply
%
% Usage:
% C = GB_spec_apply (C, Mask, accum, op, A, descriptor)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 6)
    error ('usage: C = GB_spec_apply (C, Mask, accum, op, A, descriptor)') ;
end

C = GB_spec_matrix (C) ;
A = GB_spec_matrix (A) ;
[opname optype ztype xtype] = GB_spec_operator (op, C.class) ;
[C_replace Mask_comp Atrans Btrans Mask_struct] = ...
    GB_spec_descriptor (descriptor) ;
Mask = GB_spec_getmask (Mask, Mask_struct) ;

%-------------------------------------------------------------------------------
% do the work via a clean MATLAB interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% apply the descriptor to A
if (Atrans)
    A.matrix = A.matrix.' ;
    A.pattern = A.pattern' ;
end

% T = op(A)
T.matrix = GB_spec_zeros (size (A.matrix), ztype) ;
T.pattern = A.pattern ;
T.class = ztype ;

p = T.pattern ;
x = A.matrix (p) ;

z = GB_spec_op (op, x) ;

T.matrix (p) = z ;

% C<Mask> = accum (C,T): apply the accum, then Mask, and return the result
C = GB_spec_accum_mask (C, Mask, accum, T, C_replace, Mask_comp, 0) ;

