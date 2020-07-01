function [x, y] = gb_get_pair (A)
%GB_GET_PAIR get a pair of scalars from a parameter of length 2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

type = gbtype (A) ;
desc.kind = 'full' ;
A = gbfull (A, type, 0, desc) ;
x = A (1) ;
y = A (2) ;

