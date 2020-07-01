function C = asinh (G)
%ASINH inverse hyperbolic sine.
% C = asinh (G) is the inverse hyberbolic sine of each entry G.
%
% See also GrB/sin, GrB/asin, GrB/sinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (~gb_isfloat (gbtype (G)))
    op = 'asinh.double' ;
else
    op = 'asinh' ;
end

C = GrB (gbapply (op, G)) ;

