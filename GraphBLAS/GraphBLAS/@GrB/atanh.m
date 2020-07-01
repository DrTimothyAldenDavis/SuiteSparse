function C = atanh (G)
%ATANH inverse hyperbolic tangent.
% C = atanh (G) is the inverse hyberbolic tangent of each entry G.
% C is complex if G is complex, or if any (abs (G) > 1).
%
% See also GrB/tan, GrB/atan, GrB/tanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
C = GrB (gb_trig ('atanh', G)) ;

