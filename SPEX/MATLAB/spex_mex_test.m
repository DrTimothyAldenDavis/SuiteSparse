function spex_mex_test
%SPEX_MEX_TEST run a set of tests for SPEX matlab interface
%
% Usage:  spex_mex_test
%
% See also spex_backslash, spex_lu_backslash, spex_cholesky_backslash,
%   spex_ldl_backslash, spex_mex_install, spex_mex_demo.

% Copyright (c) 2022-2026, Christopher Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

maxerr = 0 ;
rng ('default') ;

%-------------------------------------------------------------------------------
% load in the test matrices
%-------------------------------------------------------------------------------

try
    % HB/gr_30_30 matrix (problem ssget (159)):
    entries = load ('../ExampleMats/gr_30_30.mat.txt') ;
    gr_30_30 = spconvert (entries (2:end, :)) ;
    % prob = ssget(159); assert (isequal (gr_30_30, prob.A)) ;

    % HB/494_bus matrix (problem ssget (2)):
    entries = load ('../ExampleMats/494_bus.mat.txt') ;
    HB_494_bus = spconvert (entries (2:end, :)) ;
    % prob = ssget(2) ; assert (isequal (HB_494_bus, prob.A)) ;
    clear entries

catch me
    fprintf ('Error: %s\n', me.message) ;
    fprintf ('This test must be run while the SPEX/MATLAB folder\n') ;
    fprintf ('current working directory.\n') ;
    error ('Test matricies not found') ;
end

%-------------------------------------------------------------------------------
% Test SPEX Left LU with the HB/gr_30_30 matrix
%-------------------------------------------------------------------------------

fprintf ('Testing SPEX Left LU: ') ;

fprintf ('. (please wait) ') ;
A = gr_30_30 ;

m = size(A,1);
b = rand(m, 1);
fprintf ('.') ;
x = spex_lu_backslash(A,b);
x2 = A\b;
err = norm(x-x2)/norm(x);
maxerr = max (maxerr, err) ;

% now convert to an integer problem (x will not be integer)
A = floor (2^20 * A) ;
b = floor (2^20 * b) ;
fprintf ('.') ;
x = spex_lu_backslash (A, b) ;
x2 = A\b;
err = norm(x-x2)/norm(x);
maxerr = max (maxerr, err) ;
fprintf ('.') ;

%-------------------------------------------------------------------------------
% Test SPEX LU with random sparse matrices
%-------------------------------------------------------------------------------

orderings = { 'default', 'none', 'colamd', 'amd' } ;
pivotings = { 'smallest', 'diagonal', 'first', ...
    'tol smallest', 'tol largest', 'largest' } ;

for n = [1 10 100]
    for density = [0.001 0.05 0.5]

        % construct a well-conditioned problem to solve
        A = sprand(n,n,density);
        A = A+A' + n * speye (n) ;
        b = rand(n,1);

        for korder = 1:length (orderings)
            for kpiv = 1:length (pivotings)
                for tol = [0.1 0.5]

                    clear option
                    option.order = orderings {korder} ;
                    option.pivot = pivotings {kpiv} ;
                    option.tol   = tol ;

                    fprintf ('.') ;
                    x = spex_lu_backslash(A,b, option);
                    x2 = A\b;
                    err = norm(x-x2)/norm(x);
                    maxerr = max (maxerr, err) ;

                    % now convert to an integer problem (x will not be integer)
                    A = floor (2^20 * A) ;
                    b = floor (2^20 * b) ;
                    x = spex_lu_backslash(A,b, option);
                    x2 = A\b;
                    err = norm(x-x2)/norm(x);
                    maxerr = max (maxerr, err) ;
                end
            end
        end
        fprintf ('\n') ;
    end
end

fprintf ('\nmaxerr: %g\n', maxerr) ;

if (maxerr < 1e-6)
    fprintf('\nSPEX LU tests successful\n')
else
    error ('\nTesting failure!  error too high\n')
end

%-------------------------------------------------------------------------------
% Test SPEX Cholesky and LDL with HB/494_buss
%-------------------------------------------------------------------------------

fprintf ('Testing SPEX Cholesky and LDL: ') ;

% Test with the HB/494_bus matrix:
A = HB_494_bus ;
m = size(A,1);
b = rand(m, 1);
fprintf ('.') ;
x = spex_cholesky_backslash(A,b);
x1 = spex_ldl_backslash(A,b);
x2 = A\b;
err = norm(x-x2)/norm(x);
maxerr = max (maxerr, err) ;
err = norm(x-x1)/norm(x);
maxerr = max (maxerr, err) ;

% now convert to an integer problem (x will not be integer)
A = floor (2^20 * A) ;
b = floor (2^20 * b) ;
fprintf ('.') ;
x = spex_cholesky_backslash (A, b) ;
x1 = spex_ldl_backslash (A, b) ;
x2 = A\b;
err = norm(x-x1)/norm(x);
maxerr = max (maxerr, err) ;
err = norm(x-x2)/norm(x);
maxerr = max (maxerr, err) ;

% test a negative definite matrix with LDL
x3 = spex_ldl_backslash (-A, b) ;
x4 = (-A)\b;
err = norm(x3-x4)/norm(x);
maxerr = max (maxerr, err) ;
fprintf ('.') ;

%-------------------------------------------------------------------------------
% Test SPEX Cholesky and LDL with random matrices
%-------------------------------------------------------------------------------

orderings = { 'none', 'colamd', 'amd' } ;

for n = [1 10 100]
    for density = [0.001 0.05 0.5]

        % construct a well-conditioned problem to solve
        A = sprand(n,n,density);
        A = A+A' + n * speye (n) ;
        b = rand(n,1);

        for korder = 1:length (orderings)

            clear option
            option.order = orderings {korder} ;

            fprintf ('.') ;
            x = spex_cholesky_backslash(A,b, option);
            x1 = spex_ldl_backslash(A,b, option);
            x2 = A\b;
            err = norm(x-x1)/norm(x);
            maxerr = max (maxerr, err) ;
            err = norm(x-x2)/norm(x);
            maxerr = max (maxerr, err) ;

            % now convert to an integer problem (x will not be integer)
            A = floor (2^20 * A) ;
            b = floor (2^20 * b) ;
            x = spex_cholesky_backslash(A,b, option);
            x1 = spex_ldl_backslash(A,b, option);
            x2 = A\b;
            err = norm(x-x1)/norm(x);
            maxerr = max (maxerr, err) ;
            err = norm(x-x2)/norm(x);
            maxerr = max (maxerr, err) ;
        end
    end
end

fprintf ('\nmaxerr: %g\n', maxerr) ;

if (maxerr < 1e-6)
    fprintf('\nSPEX Cholesky and LDL tests successful\n')
else
    error ('spex_cholesky_backslash:test', '\nTesting failure!\n')
end

%-------------------------------------------------------------------------------
% Test SPEX Backlash with random matrices
%-------------------------------------------------------------------------------

fprintf ('Testing SPEX Backslash: ') ;
for n = [1 10 100]
    for density = [0.001 0.05 0.5]

        % construct a well-conditioned problem to solve
        A = sprand(n,n,density);
        A = A + n*speye(n);
        A2 = A+A' + n * speye (n) ;
        b = rand(n,1);

        for korder = 1:length (orderings)

            clear option
            option.order = orderings {korder} ;

            fprintf ('.') ;
            x = spex_cholesky_backslash(A2,b, option);
            x1 = spex_ldl_backslash(A2,b, option);
            x2 = A2\b;
            err = norm(x-x1)/norm(x);
            maxerr = max (maxerr, err) ;
            err = norm(x-x2)/norm(x);
            maxerr = max (maxerr, err) ;

            fprintf ('.')
            x = spex_lu_backslash(A, b, option);
            x2 = A\b;
            err = norm(x-x2)/norm(x);
            maxerr = max( maxerr, err);

            % now convert to an integer problem (x will not be integer)
            A2 = floor (2^20 * A2) ;
            b = floor (2^20 * b) ;
            x = spex_cholesky_backslash(A2,b, option);
            x1 = spex_ldl_backslash(A2,b, option);
            x2 = A2\b;
            err = norm(x-x1)/norm(x);
            maxerr = max (maxerr, err) ;
            err = norm(x-x2)/norm(x);
            maxerr = max (maxerr, err) ;

            % now convert to an integer problem (x will not be integer)
            A = floor (2^20 * A) ;
            x = spex_lu_backslash(A,b, option);
            x2 = A\b;
            err = norm(x-x2)/norm(x);
            maxerr = max (maxerr, err) ;
        end
    end
end

if (maxerr < 1e-6)
    fprintf('\nSPEX Backslash tests\n')
else
    error ('SPEX_backslash:test', '\nTesting failure!  error too high. Please reinstall\n')
end

%-------------------------------------------------------------------------------
% Test SPEX QR cases with some random matrices
%-------------------------------------------------------------------------------

fprintf ('Testing SPEX QR and SPEX Rank: ') ;

%% SPEX QR exact solutions of square full rank matrices
% If A is square and full rank, SPEX QR returns the same solution as an LU
% or Cholesky solver. In this case, it is recommended to use LU or Cholesky
% instead; however, QR can return exact solutions.
A = rand(50);
b = ones(50,1);
x = spex_qr_backslash(A,b);
x2 = spex_lu_backslash(A,b);
% This should be zero because both methods return the same solution
err = norm(x-x2);
maxerr = max (maxerr, err) ;

%% SPEX QR solving rectangular systems with m > n
% If m is larger than n, we obtain an exact least squares solution if A is
% full column rank and a basic solution otherwise. 
for n = [10 30 50 100]
    for m = [10 30 50 100]
        if (m >= n)
            fprintf ('.') ;
            A = rand(m,n); b = ones(m,1);
            x = spex_qr_backslash(A,b);
            x3 = A\b;
            err = norm(x-x3);
            maxerr = max (maxerr, err) ;
            % Now we will modify A to be rank deficient
            A(:,[2 5 7]) = ones(m,3);
            % This will return a basic solution
            x = spex_qr_backslash(A,b);
            % Backslash will not give a basic solution if A is square so only
            % do this part if A is rectangular
            if (m ~= n)
                x2 = A\b;
                err = norm(x-x2);
                maxerr = max (maxerr, err) ;
            end
        end
    end
end

%% SPEX QR solving rectangular systems with n > m
% If n is larger than m, SPEX will return a minimum norm solution. This is
% different than the solution returned by matlab backslash, however since
% it has full row rank, we do have Ax = b
for n = [10 30 50 75 100 150]
    m = ceil(n/2);
    fprintf ('.') ;
    A = rand(m,n); b = ones(m,1);
    x = spex_qr_backslash(A,b);
    err = norm(A*x-b);
    maxerr = max (maxerr, err) ;
end

%-------------------------------------------------------------------------------
% Test SPEX rank cases with some random matrices
%-------------------------------------------------------------------------------

%% SPEX Rank computing the exact rank of different matrices
% For simple well conditioned matrices, spex rank will return the same
% rank as matlab. For example, consider the following full rank and rank
% deficient matrices.
fprintf ('.') ;
A = rand(50);
r = spex_rank(A);
r2 = rank( full (A));
err = r-r2;
maxerr = max (maxerr, err) ;

fprintf ('.') ;
A = rand(25,50);
r = spex_rank(A);
r2 = rank( full (A));
err = r - r2;
maxerr = max (maxerr, err) ;

% Matrix of rank 1
fprintf ('.') ;
u = rand(100,1);
u = ceil(u*10);
A = u*u';
r = spex_rank(A);
r2 = rank( full (A));
err = r-r2;
maxerr = max (maxerr, err) ;

% However, if A is very poorly conditioned, the MATLAB rank and spex rank
% can be different.
fprintf ('.') ;
A = eye(100) - triu(ones(100), 1);
r = spex_rank(A);
r2 = rank(full (A));
if r == r2
    fprintf("\nRank calculation mistake")
end

maxerr
if (maxerr < 1e-6)
    fprintf('\nSPEX QR and Rank tests complete\n')
else
    error ('SPEX_QR and Rank:test', '\nTesting failure!  error too high. Please reinstall\n')
end

fprintf("\nAll testing complete, ready to go!\n");

