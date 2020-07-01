
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

s = [20 34 14 51] ;
Prob = ssget ('HB/west0067') ;
A = GrB (Prob.A, 'by row') ;
A = GrB (A, 'logical')

c = gap_centrality (s, A)

