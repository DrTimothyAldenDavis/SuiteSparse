function test18
% test18: test cholmod on a few large matrices
fprintf ('=================================================================\n');
fprintf ('test18: test cholmod on a few large matrices\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = UFget (887)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x = cholmod (A,b) ;
norm (A*x-b,1)

clear all

Prob = UFget (936)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x = cholmod (A,b) ;
norm (A*x-b,1)

clear all

Prob = UFget (887)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x = cholmod (A,b) ;
norm (A*x-b,1)

fprintf ('test18 passed\n') ;
