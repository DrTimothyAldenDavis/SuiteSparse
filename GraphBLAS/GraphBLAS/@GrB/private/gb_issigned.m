function s = gb_issigned (type)
%GB_ISSIGNED Determine if a type is signed or unsigned.
% s = gb_issigned (type) returns true if type is the string 'double',
% 'single', 'single complex', 'double complex', 'int8', 'int16', 'int32',
% or 'int64'.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

s = ~ (isequal (type, 'logical') || contains (type, 'uint')) ;

