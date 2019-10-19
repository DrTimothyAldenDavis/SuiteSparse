rand('state',0);
randn('state',0);
B = sprandsym(100,0.1);
B = B + 4*speye(100);

[R1,p1] = chol (B) ;
L1 = R1' ;

[L2,p2] = lchol(B) ;

p1
p2
norm (L1-L2,inf)

subplot (1,3,1) ; spy (L1) ;
subplot (1,3,2) ; spy (L2) ;
subplot (1,3,3) ; spy (B (:, 1:p1)) ;
