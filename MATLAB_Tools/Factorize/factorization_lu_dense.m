classdef factorization_lu_dense < factorization
%FACTORIZATION_LU_DENSE P*A = L*U where A is square and full.

% Copyright 2011, Timothy A. Davis, University of Florida.

    methods

        function F = factorization_lu_dense (A, fail_if_singular)
            %FACTORIZATION_LU_DENSE : P*A = L*U
            [m n] = size (A) ;
            if (m ~= n)
                error ('FACTORIZE:wrongdim', ...
                    'LU for rectangular matrices not supported.  Use QR.') ;
            end
            [f.L f.U p] = lu (A, 'vector') ;
            F.A_condest = cheap_condest (get_diag (f.U), fail_if_singular) ;
            F.A = A ;
            f.P = sparse (1:n, p, 1) ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'dense LU factorization: P*A = L*U' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using dense LU
            % x = U \ (L \ (P*b)) ;
            f = F.Factors ;
            opL.LT = true ;
            opU.UT = true ;
            x = linsolve (f.U, linsolve (f.L, full (f.P*b), opL), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense LU
            % x = (P' * (L' \ (U' \ b')))'
            f = F.Factors ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            opLT.LT = true ;
            opLT.TRANSA = true ;
            x = (f.P' * linsolve (f.L, linsolve (f.U, full (b'), opUT), opLT))';
            if (issparse (x))
                x = full (x) ;
            end
        end
    end
end
