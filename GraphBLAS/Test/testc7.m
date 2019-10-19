function testc7
%TESTC7 test complex assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default')

seed = 1 ;
for m = [1 5 10 50]
    for n = [1 5 10 50]
        seed = seed + 1 ;
        C = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        for ni = 1:m
            for nj = 1:n
                I = randperm (m, ni) ;
                J = randperm (n, nj) ;
                seed = seed + 1 ;
                A = GB_mex_random (ni, nj, 4*(ni+nj), 1, seed) ;
                C1 = C ;
                C1 (I,J) = A ;

                I0 = uint64 (I-1) ;
                J0 = uint64 (J-1) ;
                C2 = GB_mex_subassign (C, [ ], [ ], A, I0, J0, []) ;
                assert (isequal (C1, C2.matrix)) ;

                C1 = C ;
                C1 (I,J) = C1 (I,J) + A ;

                C2 = GB_mex_subassign (C, [ ], 'plus', A, I0, J0, []) ;
                assert (isequal (C1, C2.matrix)) ;

            end
        end
    end
end

fprintf ('testc7: all complex assign C(I,J)=A tests passed\n') ;

