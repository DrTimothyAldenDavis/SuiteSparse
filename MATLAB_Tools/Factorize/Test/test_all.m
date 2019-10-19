function test_all
%TEST_ALL test the Factorize package (factorize, factorize1, and inverse)
%
% If you have editted the Factorize package, type "clear classes" before
% running any tests.
%
% Example
%   test_all
%
% See also factorize, factorize1, inverse, test_factorize,
% test_performance, test_accuracy, test_disp, test_errors

% Copyright 2009, Timothy A. Davis, University of Florida

help test_all
rand ('state', 0) ;         %#ok
test_disp ;                 % test disp(F)
test_errors ;               % test error handling for invalid matrices
err1 = test_accuracy ;      % test accuracy on a range of problems
err2 = test_performance ;   % performance tests
fprintf ('\nAll tests passed, maximum error OK: %g\n', max (err1, err2)) ;

