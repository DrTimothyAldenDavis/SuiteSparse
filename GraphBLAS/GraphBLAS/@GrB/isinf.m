function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a logical matrix C where C(i,j) = true
% if G(i,j) is infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_isfloat (type) && gbnvals (G) > 0)
    C = GrB (gbapply ('isinf', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

