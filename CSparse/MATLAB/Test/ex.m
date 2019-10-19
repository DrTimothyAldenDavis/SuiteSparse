D = 10 ;
X = 1 ;
o = 0 ;

A = sparse ([
D o X o o o o X o o
o D o o X o o o o X
X o D o o o X o o o
o o o D o o o o X X
o X o o D o o o X X
o o o o o D X X o o
o o X o o X D o o o
X o o o o X o D X X
o o o X X o o X D o
o X o X X o o X o D ])

L = chol(A)'
clf
subplot (1,2,1) ; spy (A) ;
subplot (1,2,2) ; spy (L+L') ;
