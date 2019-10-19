function test17
% test17: test lchol on a few large matrices
fprintf ('=================================================================\n');
fprintf ('test17: test lchol on a few large matrices\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = UFget (887)
A = Prob.A ;
[L,s,p] = lchol (A) ;
norm (L,1)

clear all

Prob = UFget (936)
A = Prob.A ;
[L,s,p] = lchol (A) ;
norm (L,1)

clear all

Prob = UFget (887)
A = Prob.A ;
[L,s,p] = lchol (A) ;
norm (L,1)
