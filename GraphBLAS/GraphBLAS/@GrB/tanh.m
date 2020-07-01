function C = tanh (G)
%TANH hyperbolic tangent.
% C = tanh (G) is the hyperbolic tangent of each entry of G.
%
% See also GrB/tan, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
if (~gb_isfloat (gbtype (G)))
    op = 'tanh.double' ;
else
    op = 'tanh' ;
end

C = GrB (gbapply (op, G)) ;

