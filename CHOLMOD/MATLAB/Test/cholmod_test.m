function cholmod_test (nmat, no_metis)
% cholmod_test(nmat, no_metis):  test the CHOLMOD mexFunctions
%
% The UFget interface to the UF sparse matrix collection is required.
%
% nmat is optional.  If present, it is the # of matrices used in
%   tests 0, 8, 10, 11, 12, and 12.  tests 14 and 15 use 2*nmat matrices.
%   default nmat is 50.
%
% no_metis is optional.  If present, tests that require METIS are not used
%   (tests 8, 9, and 14)
%
% cholmod_demo: run tests on a few random matrices
% graph_demo: graph partitioning demo
% test0:  test most CHOLMOD functions
% test1:  test sparse2
% test2:  test sparse2
% test3:  test sparse on int8, int16, and logical
% test4:  test cholmod with multiple and sparse right-hand-sides
% test5:  test sparse2
% test6:  test sparse with large matrix, both real and complex, compare w/MATLAB
% test7:  test sparse2
% test8:  order many sparse matrices, test symbfact2, compare amd and metis
% test9:  test metis, etree, bisect, nesdis
% test10: test cholmod's backslash on real and complex matrices
% test11: test analyze, compare CHOLMOD and MATLAB, save results in Results.mat
% test12: test etree2 and compare with etree
% test13: test cholmod and MATLAB on large tridiagonal matrices
% test14: test metis, symbfact2, and etree2
% test15: test symbfact2 vs MATLAB
% test16: test cholmod on a large matrix
% test17: test lchol on a few large matrices
% test18: test cholmod on a few large matrices
% test19: look for NaN's from lchol (caused by Intel MKL 7.x bug)
% test20: test symbfact2, cholmod, and lu on a few large matrices
% test21: test cholmod on diagonal or ill-conditioned matrices
% test22: test chol and chol2 and singular and indefinite matrices
% test23: test chol and cholmod on the sparse matrix used in "bench"
% test24: test sdmult
% test25: test sdmult on a large matrix
% test26: test logical full and sparse matrices
% test27: test nesdis

% This extensive test is not included:
% test28: test nesdis

diary off
s = date ;
t = clock ;
s = sprintf ('diary cholmod_test_%s_%d-%d-%d.txt\n', s, t (4), t(5), fix(t(6)));
eval (s) ;
fprintf ('Running CHOLMOD tests.\n%s\n', s) ;

help cholmod_test

test_path = pwd ;
% addpath (test_path) ;
cd ('..') ;
cholmod_path = pwd ;
addpath (cholmod_path)
cd ('../../AMD/MATLAB') ;
amd_path = pwd ;
addpath (amd_path)
cd ('../../COLAMD') ;
colamd_path = pwd ;
addpath (colamd_path)
cd ('../CCOLAMD') ;
ccolamd_path = pwd ;
addpath (ccolamd_path)
cd ('../CAMD/MATLAB') ;
camd_path = pwd ;
addpath (camd_path)

cd (test_path)
fprintf ('Added the following paths.  You may wish to add them\n') ;
fprintf ('permanently using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', cholmod_path) ;
fprintf ('%s\n', amd_path) ;
fprintf ('%s\n', colamd_path) ;
fprintf ('%s\n', ccolamd_path) ;
fprintf ('%s\n', camd_path) ;

input ('\n\n--------- Hit enter to contine: ') ;

if (nargin < 1)
    nmat = 50 ;
end

do_metis = (nargin < 2) ;

cholmod_demo
graph_demo
test0 (nmat)
test1
test2
test3
test4
test5
test6
test7

if (do_metis)
    % these tests require METIS
    test8 (nmat)
    test9
end

test10 (nmat)
test11 (nmat)
test12 (nmat)
test13

if (do_metis)
    % this test requires METIS
    test14 (2*nmat)
end

test15 (2*nmat)
test16
test17
test18
test19
test20
test21
test22 (nmat)
test22 (0)
test23
test24
test25
test26 (do_metis)
test27
% test28

fprintf ('=================================================================\n');
fprintf ('all tests passed\n') ;

diary off
