classdef factorization_dense_lu < factorization_generic
%FACTORIZATION_DENSE_LU dense LU factorization: p*A = L*U

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_dense_lu (A)
            %FACTORIZATION_DENSE_LU dense LU: p*A = L*U
            n = size (A,2) ;
            [F.L U p] = lu (A, 'vector') ;
            assert (nnz (diag (U)) == n, 'Matrix is rank deficient.') ;
            F.p = sparse (1:n, p, 1) ;
            F.U = U ;
            F.A = A ;
        end

        function disp (F)
            %DISP displays a dense LU factorization
            fprintf ('  dense LU factorization: p*A = L*U\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  L:\n') ; disp (F.L) ;
            fprintf ('  U:\n') ; disp (F.U) ;
            fprintf ('  p:\n') ; disp (F.p) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using dense LU
            % x = U \ (L \ (p*b)) ;
            if (issparse (b))
                b = full (b) ;
            end
            opL.LT = true ;
            opU.UT = true ;
            x = linsolve (F.U, linsolve (F.L, F.p*b, opL), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense LU
            % x = (p' * (L' \ (U' \ b')))'
            bT = b' ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            opLT.LT = true ;
            opLT.TRANSA = true ;
            x = (F.p' * linsolve (F.L, linsolve (F.U, bT, opUT), opLT))' ;
        end
    end
end
