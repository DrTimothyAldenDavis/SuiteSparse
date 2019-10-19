function err = test_factorize (A)
%TEST_FACTORIZE test the accuracy of the factorization object
%
% Example
%   test_factorize (A) ;    % where A is a square matrix (sparse or dense)
%
% See also test_all, factorize1, factorize, inverse, mldivide

% Copyright 2009, Timothy A. Davis, University of Florida

if (nargin < 1)
    A = rand (100) ;
end

[m n] = size (A) ;
err = 0 ;
if (min (m,n) > 0)
    anorm = norm (A,1) ;
else
    anorm = 1 ;
end

for nrhs = 1:3

    for bsparse = 0:1

        % fprintf ('n %4d nrhs %d bsparse %d : ', n, nrhs, bsparse) ;

        b = rand (m,nrhs) ;
        if (bsparse)
            b = sparse (b) ;
        end

        %-------------------------------------------------------------------
        % test backslash and related methods
        %-------------------------------------------------------------------

        % method 0:
        x = A\b ;           err = check_resid (err, anorm, A, x, b) ;

        % method 1:
        S = inverse (A) ;
        x = S*b ;           err = check_resid (err, anorm, A, x, b) ;

        % method 2:
        if (m == n)
            F = factorize1 (A) ;
            x = F\b ;       err = check_resid (err, anorm, A, x, b) ;
        end

        % method 3:
        if (m == n)
            S = inverse (F) ;
            x = S*b ;       err = check_resid (err, anorm, A, x, b) ;
        end

        % method 4:
        F = factorize (A) ;
        x = F\b ;           err = check_resid (err, anorm, A, x, b) ;

        % method 5:
        S = inverse (F) ;
        x = S*b ;           err = check_resid (err, anorm, A, x, b) ;

        % method 6:
        if (m == n)
            [L,U,p] = lu (A, 'vector') ;
            x = U \ (L \ (b (p,:))) ;
            err = check_resid (err, anorm, A, x, b) ;
        end

        % method 7 (ack!)
        if (m == n)
            S = inv (A) ;
        else
            S = pinv (full (A)) ;
            if (issparse (A))
                S = sparse (S) ;
            end
        end
        x = S*b ;           err = check_resid (err, anorm, A, x, b) ;

        %-------------------------------------------------------------------
        % test mtimes
        %-------------------------------------------------------------------

        S = inverse (F) ;
        d = rand (n,1) ;
        x = F*d ;
        y = A*d ;
        z = S\d ;
        e = max (norm (x-y,1), norm (x-z,1)) ;
        if (e > 0)
            error ('mtimes error') ;
        end

        if (m == n)
            F = factorize1 (A) ;
            S = inverse (F) ;
            d = rand (n,1) ;
            x = F*d ;
            y = A*d ;
            z = S\d ;
            e = max (norm (x-y,1), norm (x-z,1)) ;
            if (e > 0)
                error ('mtimes error') ;
            end
        end

        %-------------------------------------------------------------------
        % test slash and related methods
        %-------------------------------------------------------------------

        b = rand (nrhs,n) ;
        if (bsparse)
            b = sparse (b) ;
        end

        % method 0:
        x = b/A ;           err = check_resid (err, anorm, A, x, b, 1) ;

        % method 1:
        S = inverse (A) ;
        x = b*S ;           err = check_resid (err, anorm, A, x, b, 1) ;

        % method 2:
        if (m == n)
            F = factorize1 (A) ;
            x = b/F ;       err = check_resid (err, anorm, A, x, b, 1) ;
        end

        % method 3:
        if (m == n)
            S = inverse (F) ;
            x = b*S ;       err = check_resid (err, anorm, A, x, b, 1) ;
        end

        % method 4:
        F = factorize (A) ;
        x = b/F ;           err = check_resid (err, anorm, A, x, b, 1) ;

        % method 5:
        S = inverse (F) ;
        x = b*S ;           err = check_resid (err, anorm, A, x, b, 1) ;

        % method 6:
        if (m == n)
            [L,U,p] = lu (A, 'vector') ;
            x = (b / U) / L ; x (:,p) = x ;
            err = check_resid (err, anorm, A, x, b, 1) ;
        end

        % method 7 (ack!)
        if (m == n)
            S = inv (A) ;
        else
            S = pinv (full (A)) ;
            if (issparse (A))
                S = sparse (S) ;
            end
        end
        x = b*S ;           err = check_resid (err, anorm, A, x, b, 1) ;

        %-------------------------------------------------------------------
        % test double
        %-------------------------------------------------------------------

        Y = double (inverse (A)) ;
        if (m == n)
            Z = inv (A) ;
        else
            Z = pinv (full (A)) ;
        end
        e = norm (Y-Z,1) ;
        if (n > 0)
            e = e / norm (Z,1) ;
        end
        err = max (e, err) ;

        %-------------------------------------------------------------------
        % test subsref
        %-------------------------------------------------------------------

        F = factorize (A) ;
        Y = inverse (A) ;
        if (numel (A) > 1)
            if (F (end) ~= A (end))
                error ('factorization subsref error') ;
            end
            if (F.A (end) ~= A (end))
                error ('factorization subsref error') ;
            end
        end
        if (n > 0)
            if (F (1,1) ~= A (1,1))
                error ('factorization subsref error') ;
            end
            if (F.A (1,1) ~= A (1,1))
                error ('factorization subsref error') ;
            end
            e = abs (Y (1,1) - Z (1,1)) ;
            err = max (e,err) ;
            if (m > 1 && n > 1)
                e = norm (Y (1:2,1:2) - Z (1:2,1:2), 1) ;
                err = max (e,err) ;
            end
        end
        if (m > 3 && n > 1)
            if (any (F (2:end, 1:2) - A (2:end, 1:2)))
                error ('factorization subsref error') ;
            end
            if (any (F (2:4, :) - A (2:4, :)))
                error ('factorization subsref error') ;
            end
            if (any (F (:, 1:2) - A (:, 1:2)))
                error ('factorization subsref error') ;
            end
        end

        %-------------------------------------------------------------------
        % test update/downdate
        %-------------------------------------------------------------------

        if (F.kind == 6)
            w = rand (n,1) ;
            b = rand (n,1) ;
            % update
            G = F + w ;
            x = G\b ;       err = check_resid (err, anorm, A+w*w', x, b) ;
            % downdate
            G = G - w ;
            x = G\b ;       err = check_resid (err, anorm, A, x, b) ;
            clear G
        end

        %-------------------------------------------------------------------
        % test size
        %-------------------------------------------------------------------

        [m1 n1] = size (F) ;
        [m n] = size (A) ;
        if (m1 ~= m || n1 ~= n)
            error ('size error') ;
        end
        [m1 n1] = size (Y) ;
        if (m1 ~= n || n1 ~= m)
            error ('pinv size error') ;
        end
        if (size (Y,1) ~= n || size (Y,2) ~= m)
            error ('pinv size error') ;
        end
        if (size (F,1) ~= m || size (F,2) ~= n)
            error ('size error') ;
        end

        %-------------------------------------------------------------------
        % test mtimes
        %-------------------------------------------------------------------

        d = rand (1,m) ;
        x = d*F ;
        y = d*A ;
        z = d/Y ;
        e = max (norm (x-y,1), norm (x-z,1)) ;
        if (e > 0)
            error ('mtimes error') ;
        end

        if (m == n)
            F = factorize1 (A) ;
            Y = inverse (F) ;
            d = rand (1,m) ;
            x = d*F ;
            y = d*A ;
            z = d/Y ;
            e = max (norm (x-y,1), norm (x-z,1)) ;
            if (e > 0)
                error ('mtimes error') ;
            end
        end

        %-------------------------------------------------------------------
        % test inverse
        %-------------------------------------------------------------------

        Y = double (inverse (inverse (A))) ;
        e = norm (A-Y,1) ;
        if (e > 0)
            error ('inverse error') ;
        end

    end
end

fprintf ('.') ;

if (err > 1e-8)
    fprintf ('error: %8.3e\n', err) ;
    error ('error is too high!') ;
end

%---------------------------------------------------------------------------

function err = check_resid (err, anorm, A, x, b, transposed)
if (nargin < 6)
    transposed = 0 ;
end
[m n] = size (A) ;

if (transposed)
    if (m >= n)
        e = norm (A'*x'-b',1) / (anorm + norm (x,1)) ;
    else
        e = norm (A*(A'*x')-A*b',1) / (anorm + norm (x,1)) ;
    end
else
    if (m <= n)
        e = norm (A*x-b,1) / (anorm + norm (x,1)) ;
    else
        e = norm (A'*(A*x)-A'*b,1) / (anorm + norm (x,1)) ;
    end
end

if (min (m,n) > 1)
    if (issparse (A) && issparse (b))
        if (~issparse (x))
            error ('x must be sparse') ;
        end
    else
        if (issparse (x))
            error ('x must be full') ;
        end
    end
end
err = max (err, e) ;

