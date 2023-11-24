
% clear all
% clear mex
% format compact

fprintf ('testing a small problem:\n') ;
Prob = ssget (1440)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x1 = A\b ;
err1 = norm (A*x1-b)
x2 = cholmod2 (A,b) ;
err2 = norm (A*x2-b)
x3 = cholmod2 (A,b, 'single') ;
err3 = norm (A*double(x3)-b)
whos

fprintf ('testing a large problem (nd12k):\n') ;

Prob = ssget (938)
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
I = speye (n) ;
% condest (A) is 2e7 so single precision has trouble;
% so add 100*I:
A = A + 100*I ;

tic
x1 = A\b ;
t = toc ;
fprintf ('time for x=A\\b (in double):        %g sec\n', t) ;

tic
x2 = cholmod2 (A, b) ;
t = toc ;
fprintf ('time for x=cholmod2(A,b) (double): %g sec\n', t) ;

fprintf ('... please wait ...\n') ;
b2 = single (b) ;
tic
x3 = cholmod2 (A, b2, 'single') ;
t = toc ;
fprintf ('time for x=cholmod2(A,b) (single): %g sec\n', t) ;

anorm = norm (A,1)
norm (A*x1-b,1) / anorm
norm (A*x2-b,1) / anorm
norm (A*double (x3) - b, 1) / anorm

