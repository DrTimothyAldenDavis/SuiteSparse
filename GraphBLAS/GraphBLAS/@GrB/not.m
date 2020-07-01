function C = not (G)
%~ logical negation.
% C = ~G computes the logical negation of G.  The result C is full.
% To negate just the entries in the pattern of G, use
% C = GrB.apply ('~.logical', G), which has the same pattern as G.
%
% See also GrB.apply.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
C = GrB (gbapply ('~', gbfull (G, 'logical'))) ;

