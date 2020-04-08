
GrB.burble (1)
A = GrB (ones (3))
% C = A*A
x = GrB ([1 2 3]')
% C

C1 = GrB.mxm (A, '+.*.double', x)

A = single (A)
x = single (x)
'max:2nd'
C2 = GrB.mxm (A, 'max.second.single', x)
%{
A = logical (A)
x = logical (x)
C = GrB.mxm (A, '|.&.logical', x)
%}





