function s = isdiag (G)
%ISDIAG true if G is a diagonal matrix.
% isdiag (G) is true if G is a diagonal matrix, and false otherwise.
%
% See also GrB/isbanded.

% FUTURE: this will faster when 'gb_bandwidth' is a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

% using gb_bandwidth instead:
% [lo, hi] = gb_bandwidth (G) ;
% s = (lo == 0) & (hi == 0) ;

s = (gbnvals (gbselect ('diag', G, 0)) == gbnvals (G)) ;

