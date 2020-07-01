function s = gb_isfloat (type)
%GB_ISFLOAT true for floating-point GraphBLAS types.
% Implements s = isfloat (type (G))

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

s = contains (type, 'double') || contains (type, 'single') ;

