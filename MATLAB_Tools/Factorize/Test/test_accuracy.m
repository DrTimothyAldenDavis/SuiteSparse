function err = test_accuracy
%TEST_ACCURACY test the accuracy of the factorize object
%
% Example
%   err = test_accuracy
%
% See also test_all, test_factorize.

% Copyright 2011, Timothy A. Davis, University of Florida.

fprintf ('\nTesting accuracy:\n') ;
reset_rand ;

A = [ 0.1482    0.3952    0.1783    1.1601
      0.3952    0.3784    0.2811    0.4893
      0.1783    0.2811    1.1978    1.3837
      1.1601    0.4893    1.3837    0.7520 ] ;

F = factorize (A, 'ldl', 1) ;                                               %#ok
err = test_factorize (sparse (A)) ;
err = max (err, test_factorize (A)) ;

% dense matrices
% err = 0 ;
for n = 0:6
    for im = 0:1
        A = rand (n) ;
        if (im == 1)
            A = A + 1i * rand (n) ;
        end
        % unsymmetric
        err = max (err, test_factorize (A)) ;
        % dense, symmetric but not always positive definite
        A = A+A' ;
        err = max (err, test_factorize (A)) ;
        % symmetric positive definite
        A = A'*A + eye (n) ;
        err = max (err, test_factorize (A)) ;
        % least-squares problem
        A = rand (2*n,n) ;
        err = max (err, test_factorize (A)) ;
        % under-determined problem
        A = A' ;
        err = max (err, test_factorize (A)) ;
    end
    fprintf ('\n') ;
end
% default dense 100-by-100 matrix
err = max (err, test_factorize) ;

fprintf ('\nerr so far: %g\nplease wait ', err) ;

for im = 0:1

    % sparse rectangular
    A = sprandn (5,10,0.6) + speye (5,10) ;
    if (im == 1)
        A = A + 1i * sprandn (5,10,0.2) ;
    end
    err = max (err, test_factorize (A)) ;
    A = A' ;
    err = max (err, test_factorize (A)) ;

    % sparse, unsymmetric
    load west0479
    A = west0479 ;
    if (im == 1)
        A = A + 1i * sprand (A) ;
    end
    err = max (err, test_factorize (A)) ;

    % sparse, symmetric, but not positive definite
    A = abs (A+A') + eps * speye (size (A,1)) ;
    err = max (err, test_factorize (A)) ;

    % sparse symmetric positive definite
    A = delsq (numgrid ('L', 8)) ;
    if (im == 1)
        A = A + 1i * (sprand (A) + sprand (A)') ;
    end
    err = max (err, test_factorize (A)) ;

end

if (err > 1e-6)
    error ('error to high!  %g\n', err) ;
end

fprintf ('\nmax error is OK: %g\n', err) ;
