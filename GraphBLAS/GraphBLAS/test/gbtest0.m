function gbtest0
%GBTEST0 test GrB.clear

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

GrB.clear

assert (isequal (GrB.format, 'by col')) ;
assert (isequal (GrB.chunk, 64*1024)) ;

GrB.burble (1) ;
GrB.burble (0) ;
assert (~GrB.burble) ;

GrB.burble (false) ;
assert (~GrB.burble) ;

ok = true ;
try
    GrB.burble (rand (2)) ;
    ok = false ;
catch me
end
assert (ok) ;

fprintf ('default # of threads: %d\n', GrB.threads) ;

fprintf ('gbtest0: all tests passed\n') ;

