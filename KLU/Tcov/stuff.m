function B = stuff (A, newsize)
% B = stuff (A, newsize)
%
% compresses pattern of A into B, of size newsize
% newsize defaults to 200

if (nargin < 2)
    newsize = 200 ;
end

[m n] = size (A) ;
if (m ~= n)
    error ('Square matrices only') ;
end

if (n < newsize)
    B = spones (A) ;
    return
end

[i j] = find (A) ;

ratio = newsize / n ;

i = floor (1 + (i - 1) * ratio) ;
j = floor (1 + (j - 1) * ratio) ;
x = ones (1, length (i)) ;

B = sparse (i,j,x, newsize, newsize) ;
