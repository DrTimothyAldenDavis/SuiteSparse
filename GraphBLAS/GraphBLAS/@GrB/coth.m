function C = coth (G)
%COTH hyperbolic cotangent.
% C = coth (G) is the hyperbolic cotangent of each entry of G.
% Since coth (0) is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'tanh.double' ;
else
    op = 'tanh' ;
end

C = GrB (gbapply ('minv', gbfull (gbapply (op, G)))) ;

