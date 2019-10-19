% function test2
% test2: test sparse2
fprintf ('=================================================================\n');
fprintf ('test2: test sparse2\n') ;

i = [ 2 3 ]
j = [ 3 4 ]
s = [11.4 9.2] + 1i * [3.4 1.2]
sparse (i,j,s)
sparse2 (i,j,s)

n = 100 ;
nz = 4000 ;

i = fix (n * rand (nz,1)) + 1 ;
j = fix (n * rand (nz,1)) + 1 ;
s = rand (nz,1) + 1i * rand (nz,1) ;
A = sparse (i,j,s,n,n) ;
B = sparse2 (i,j,s,n,n) ;
nnz(A)

if (norm (A-B,1) > 1e-14)
    A_minus_B = A-B
    error ('!') ;
end

C = sparse (A) ;
D = sparse2 (B) ;

if (norm (C-D,1) > 1e-14)
    C_minus_D = C-D
    error ('!') ;
end
% spy(C)

fprintf ('test2 passed\n') ;
