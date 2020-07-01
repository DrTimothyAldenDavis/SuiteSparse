function X = nonzeros (G)
%NONZEROS extract entries from a matrix.
% X = nonzeros (G) extracts the entries from G.  X has the same type as G
% ('double', 'single', 'int8', ...).  If G contains explicit entries with a
% value of zero, these are dropped from X.  To return those entries, use
% [I,J,X] = GrB.extracttuples (G).  This function returns the X of
% [I,J,X] = find (G), which also drops explicit zeros.
%
% See also GrB.extracttuples, GrB.entries, GrB.nonz, GrB/find.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
X = gbextractvalues (gbselect ('nonzero', G)) ;

