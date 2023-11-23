
clear all
clear mex
format compact
Prob = ssget (1440)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x1 = A\b ;
norm (A*x1-b)
x2 = cholmod2 (A,b)
norm (A*x2-b)

x3 = cholmod2 (A,b, 'single')
whos
norm (A*double(x3)-b)

Prob = ssget (938) ;
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
I = speye (n) ;
% condest (A) is 2e7 so single precision has trouble;
% so add 100*I:
A = A + 100*I ;

tic
x1 = A\b ;
toc

tic
x2 = cholmod2 (A, b) ;
toc

b2 = single (b) ;
tic
x3 = cholmod2 (A, b2, 'single') ;
toc

anorm = norm (A,1)
norm (A*x1-b,1) / anorm
norm (A*x2-b,1) / anorm
norm (A*double (x3) - b, 1) / anorm

