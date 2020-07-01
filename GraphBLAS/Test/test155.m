function test155
%TEST155 test GrB_*_setElement and GrB_*_removeElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
mlist  = [ 1  1  10  20 ] ;
nlist  = [ 1 10   1  10 ] ;
nzlist = [ 5 100 100 1000 ] ;

for trial = 1:4

    % fprintf ('trial: %d\n', trial) ;
    m = mlist (trial) ;
    n = nlist (trial) ;
    nz = nzlist (trial) ;
    I = irand (1, m, nz, 1) ;
    J = irand (1, n, nz, 1) ;
    X = rand (nz, 1) ;
    Action = double (rand (nz, 1) > 0.4) ;

    % do the work in MATLAB
    C1 = sparse (m, n) ;
    for k = 1:nz
        if (Action (k) == 0)
            C1 (I (k), J (k)) = sparse (0) ;
        else
            C1 (I (k), J (k)) = sparse (X (k)) ;
        end
    end

    % do the work in GraphBLAS (default input matrix)
    C2 = GB_mex_edit (sparse (m, n), I, J, X, Action) ;
    assert (isequal (C1, C2)) ;

    % do the work in GraphBLAS (all hyper / csc/csr cases)
    clear C0
    for is_hyper = 0:1
        for is_csc = 0:1
            C0.matrix = sparse (m, n) ;
            C0.is_hyper = is_hyper ;
            C0.is_csc = is_csc ;
            C2 = GB_mex_edit (C0, I, J, X, Action) ;
            assert (isequal (C1, C2)) ;
        end
    end
end

fprintf ('test155: all tests passed\n') ;

