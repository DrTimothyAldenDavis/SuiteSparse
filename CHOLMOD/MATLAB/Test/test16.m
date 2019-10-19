function test16
% test16: test cholmod on a large matrix

fprintf ('=================================================================\n');
fprintf ('test16: test cholmod on a large matrix\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = UFget (936)
A = Prob.A ;
% tic
% [L,s,p] = lchol (A) ;
% toc
% norm (L,1)

n = size (A,1) ;
b = rand (n,1) ;
tic
x = cholmod(A,b) ;
t = toc
err = norm (A*x-b) ;

if (err > 1e-5)
    error ('!') ;
end

fprintf ('test16 passed\n') ;
