function test7
% test7: test sparse2
fprintf ('=================================================================\n');
fprintf ('test7: test sparse2\n') ;

randn ('state', 0) ;
rand  ('state', 0) ;

% Prob = UFget (437)
Prob = UFget (750)
A = Prob.A ;
[m n] = size (A) ;

tic
[i j x] = find (A) ;
t = toc ;
fprintf ('find time %8.4f\n', t) ;

tic ;
B = sparse2 (i,j,x,m,n) ;
t1 = toc ;
fprintf ('tot: %8.6f\n', t1)

tic ;
B = sparse2 (i,j,x,m,n) ;
t1 = toc ;
fprintf ('tot: %8.6f again \n', t1) ;

tic ;
B1 = sparse2 (i,j,x) ;
t1 = toc ;
fprintf ('tot: %8.6f (i,j,x)\n', t1) ;

nz = length (x) ;
p = randperm (nz) ;

i2 = i(p) ;
j2 = j(p) ;
x2 = x(p) ;

tic ;
B = sparse2 (i,j,x,m,n) ;
t1 = toc ;

fprintf ('tot: %8.6f  (jumbled)\n', t1) ;

ii = [i2 ; i2] ;
jj = [j2 ; j2] ;
xx = rand (2*nz,1) ;

tic ;
D = sparse2 (ii,jj,xx,m,n) ;
t1 = toc ;

fprintf ('tot %8.6f  (duplicates)\n', t1) ;

fprintf ('test7 passed\n') ;
