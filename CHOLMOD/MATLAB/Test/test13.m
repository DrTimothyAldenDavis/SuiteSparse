function test13
% test cholmod and MATLAB on large tridiagonal matrices
fprintf ('=================================================================\n');
fprintf ('test13: test cholmod and MATLAB on large tridiagonal matrices\n') ;

for n = [10000 1e4 1e5 1e6]

    e = ones (n,1) ;
    A = spdiags ([e 4*e e], -1:1, n, n) ;
    clear e
    b = rand (n,1) ;
    pack

    tic ;
    x = cholmod (A,b) ;
    t2 = toc ;
    e = norm (A*x-b,1) ;
    fprintf ('n %9d   cholmod %8.2f  err %6.1e\n', n, t2, e) ;

    tic ;
    x = A\b ;
    t1 = toc ;
    e = norm (A*x-b,1) ;
    fprintf ('n %9d   matlab  %8.2f  err %6.1e\n', n, t1, e) ;

    clear A b
    pack

end


