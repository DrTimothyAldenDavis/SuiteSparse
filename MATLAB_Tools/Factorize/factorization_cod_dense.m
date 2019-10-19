classdef factorization_cod_dense < factorization
%FACTORIZATION_COD_DENSE complete orthogonal factorization: A = U*R*V' where A is full.
% A fairly accurate estimate of rank is found.  double(inverse(F)) is a fairly
% accurate estimate of pinv(A).

% Copyright 2011, Timothy A. Davis, University of Florida.

    methods

        function F = factorization_cod_dense (A)
            %FACTORIZATION_COD_DENSE A = U*R*V'
            [f.U f.R f.V F.A_rank] = cod (A) ;
            F.A = A ;
            F.Factors = f ;
            F.kind = 'dense COD factorization: A = U*R*V''' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using a dense COD factorization
            % x = V * (R \ (U' * b))
            f = F.Factors ;
            x = f.V * linsolve (f.R, full (f.U' * b), struct ('UT', true)) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense COD factorization
            % x = ((b * V) / R) * U' = (U * (R' \ (b*V)'))'
            f = F.Factors ;
            op.UT = true ;
            op.TRANSA = true ;
            x = (f.U * linsolve (f.R, full ((b * f.V)'), op))' ;
        end
    end
end
