classdef factorization_ldl_dense < factorization
%FACTORIZATION_LDL_DENSE P'*A*P = L*D*L' where A is sparse and full

% Copyright 2011, Timothy A. Davis, University of Florida.

    methods

        function F = factorization_ldl_dense (A)
            %FACTORIZATION_LDL_DENSE : P'*A*P = L*D*L'
            if (isempty (A))
                f.L = [ ] ;
                f.D = [ ] ;
                P = [ ] ;
            else
                [f.L f.D P] = ldl (A, 'vector') ;
            end
            f.D = sparse (f.D) ;    % D is block diagonal with 2-by-2 blocks
            c = full (condest (f.D)) ;
            if (c > 1/(2*eps))
                error ('MATLAB:singularMatrix', ...
                    'Matrix is singular to working precision.') ;
            end
            n = size (A,1) ;
            f.P = sparse (P, 1:n, 1) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.A_condest = c ;
            F.kind = 'dense LDL factorization: P''*A*P = L*D*L''' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using dense LDL
            % x = P * (L' \ (L \ (P' * b)))
            f = F.Factors ;
            x = f.P * (f.L' \ (f.D \ (f.L \ (f.P' * b)))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense LDL
            % x = (P * (L' \ (L \ (P' * b'))))'
            x = (mldivide_subclass (F, b'))' ;
        end
    end
end
