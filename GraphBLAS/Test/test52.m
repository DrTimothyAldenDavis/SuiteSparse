function test52
%TEST52 test AdotB vs AxB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('\n----------------------- AdotB versus AxB\n') ;

rng ('default')
x = sparse (rand (10,1)) ;
y = sparse (rand (10,1)) ;
c = x'*y ;
'AdotB'
C = GB_mex_AdotB (x, y)
'did AdotB'
c - C
assert (isequal (c, C))

x = sprandn (100, 1, 0.2) ;
y = sprandn (100, 1, 0.2) ;
c = x'*y ;
C = GB_mex_AdotB (x, y)
assert (isequal (c, C))

for m = 1:10
    for n = 1:10
        for k = [10 100 1000 10000]

            A = sprandn (k, m, 0.2) ;
            B = sprandn (k, n, 0.2) ;
            Mask = sprand (m, n, 0.5) ;

            C = A'*B ;
            C2 = GB_mex_AdotB (A,B) ;

            assert (isequal (C, C2)) ;
            assert (spok (C2) == 1)

            C = spones (Mask) .* C ;
            C2 = GB_mex_AdotB (A,B, Mask) ;

            assert (isequal (C, C2)) ;
            assert (spok (C2) == 1)
        end
    end
end

k = 10e6 ;
fprintf ('\nbuilding random sparse matrices %d by M\n', k) ;
for m = 1:20
    A = sprandn (k, m, 0.1) ;
    B = sprandn (k, m, 0.1) ;
    Mask = spones (sprandn (m, m, 0.5)) ;

    % fprintf ('MATLAB:\n') ;
    tic
    C = A'*B ;
    t1 = toc ;

    % fprintf ('GrB AdotB:\n') ;
    tic
    C2 = GB_mex_AdotB (A,B) ;
    t2 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C4 = GB_mex_AxB (A,B, true) ;
    t4 = toc ;

    fprintf (...
    'm %2d MATLAB: %10.4f AdotB : %10.4f   GB,auto:: %10.4f', ...
    m, t1, t2, t4) ;
    fprintf (' speedup: %10.4f (no Mask)\n', t1/t4) ;

    assert (isequal (C, C2)) ;
    assert (isequal (C, C4)) ;

    % fprintf ('MATLAB:\n') ;
    tic
    C = spones (Mask) .* (A'*B) ;
    t1 = toc ;

    % fprintf ('GrB AdotB:\n') ;
    tic
    C2 = GB_mex_AdotB (A,B, Mask) ;
    t2 = toc ;

    % fprintf ('GrB A''*B native:\n') ;
    tic
    C4 = spones (Mask) .* GB_mex_AxB (A,B, true) ;
    t4 = toc ;

    fprintf (...
    'm %2d MATLAB: %10.4f AdotB : %10.4f   GB,auto:: %10.4f', ...
    m, t1, t2, t4) ;
    fprintf (' speedup: %10.4f (with Mask)\n', t1/t4) ;

    assert (isequal (C, C2)) ;
    assert (isequal (C, C4)) ;

end

k = 30e6
fprintf ('building random sparse matrix, %d by %d\n', k,2) ;
A = sprandn (k, 2, 0.01) ;
B = sprandn (k, 2, 0.01) ;

fprintf ('MATLAB:\n') ;
tic
C = A'*B ;
toc

fprintf ('GrB AdotB:\n') ;
tic
C2 = GB_mex_AdotB (A,B) ;
toc

fprintf ('GrB (A'')*B:\n') ;
tic
C3 = GB_mex_AxB (A',B) ;
toc

fprintf ('GrB A''*B native:\n') ;
tic
C4 = GB_mex_AxB (A,B, true) ;
toc

assert (isequal (C, C2)) ;
assert (isequal (C, C3)) ;
assert (isequal (C, C4)) ;


k = 30e6
m = 100
fprintf ('building random sparse matrix, %d by %d\n', k,m) ;
A = sprandn (k, 2, 0.01) ;
B = sprandn (k, m, 0.01) ;

fprintf ('MATLAB:\n') ;
tic
C = A'*B ;
toc

fprintf ('GrB AdotB:\n') ;
tic
C2 = GB_mex_AdotB (A,B) ;
toc

fprintf ('GrB (A'')*B:\n') ;
tic
C3 = GB_mex_AxB (A',B) ;
toc

fprintf ('GrB A''*B native:\n') ;
tic
C4 = GB_mex_AxB (A,B, true) ;
toc

assert (isequal (C, C2)) ;
assert (isequal (C, C3)) ;
assert (isequal (C, C4)) ;

fprintf ('\ntest52: all tests passed\n') ;

