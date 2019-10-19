function test_errors
%TEST_ERRORS tests error handling for the factorize object methods
%
% Example
%   test_errors
%
% See also test_all, factorize1, factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

% the matrix A must be square for factorize1
fprintf ('\nTesting error-handling, error messages expected:\n') ;
ok = true ;
A = rand (3,2) ;
try
    F = factorize1 (A) ;                                               %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% the matrix A must be 2D for factorize
A = ones (2,2,2) ; 
try
    F = factorize (A) ;                                                %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% the matrix A must be full-rank
A = ones (3) ;
try
    F = factorize (A) ;                                                %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

try
    F = factorize1 (A) ;                                               %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

A = ones (3,2) ;
try
    F = factorize (A) ;                                                %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

A = ones (2,3) ;
try
    F = factorize (A) ;                                                %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot use cell indexing
A = rand (3,2) ;
F = factorize (A) ;
try
    C = F {1} ;                                                        %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot use linear indexing of the inverse
S = inverse (A) ;
try
    C = S (1) ;                                                        %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% invalid indexing
try
    C = F (1,1).L ;                                                    %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% invalid indexing
try
    C = F.L (1,1).stuff ;                                              %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% non-existent field
try
    C = F.junk ;                                                       %#ok
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% can only update/downdate a dense Cholesky factorization
A = rand (2) ;
F = factorize (A) ;
w = rand (2,1) ;
try
    F = F + w ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end
try
    F = F - w ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

if (~ok)
    error ('error-handling failed') ;
end
fprintf ('\nAll error-handing tests passed\n') ;
