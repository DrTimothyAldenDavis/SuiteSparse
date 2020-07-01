function C = angle (G)
%ANGLE phase angle.
% C = angle (G) is the phase angle of each entry of G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (contains (type, 'complex'))
    C = GrB (gbapply ('carg', G)) ;
else
    % C is all zero
    C = GrB (gbnew (m, n, type)) ;
end

