function b = rhs (m)
% b = rhs (m), compute a right-hand-side
b = ones (m,1) + (0:m-1)'/m ;
