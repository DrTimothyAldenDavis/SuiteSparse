function err = test_accuracy
%TEST_ACCURACY test the accuracy of the factorize object
%
% Example
%   err = test_accuracy
%
% See also test_all, test_factorize.

% Coyright 2009, Timothy A. Davis, University of Florida

fprintf ('\nTesting accuracy: ') ;

% dense matrices
err = 0 ;
for n = 0:20
    A = rand (n) ;
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
% default dense 100-by-100 matrix
err = max (err, test_factorize) ;

fprintf (' please wait ') ;

% sparse rectangular
A = sprandn (5,10,0.6) + speye (5,10) ;
err = max (err, test_factorize (A)) ;
A = A' ;
err = max (err, test_factorize (A)) ;

% sparse, unsymmetric
load west0479
A = west0479 ;
err = max (err, test_factorize (A)) ;

% sparse, symmetric, but not positive definite
A = abs (A+A') + eps * speye (size (A,1)) ;
err = max (err, test_factorize (A)) ;

% sparse symmetric positive definite
A = delsq (numgrid ('L', 15)) ;
err = max (err, test_factorize (A)) ;

fprintf ('\nmax error is OK: %g\n', err) ;

