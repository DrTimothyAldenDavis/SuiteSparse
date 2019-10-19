
function write (A,b)


[m n] = size (A) ;

if (nargin < 2)
    b = rand (m,1) ;
end

opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;

[mb nrhs] = size (b) ;

if (m ~= n)
    error ('A must be square') ;
end
if (mb ~= m)
    error ('A and b must have same # rows') ;
end

% column pointers
p = [0 (cumsum (full (sum (spones (A)))))]  ;

% row indices and numerical values
[row_indices j values] = find (A) ;

nz = nnz (A) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
%fprintf (options, '%s\n', 'default') ;
fprintf (options, '%g\n', opts) ;
fclose (options) ;

f = fopen ('Afile', 'w') ;
fprintf (f, '%s\n', 'n, nnz, real, nrhs, isRHSreal') ;
fprintf (f, '%d\n%d\n%d\n', n, nz, isreal(A), nrhs, isreal(b)) ;
fprintf (f, '%s\n', 'column pointers') ;
fprintf (f, '%d\n', p) ;
fprintf (f, '%s\n', 'row indices') ;
fprintf (f, ' %d\n', row_indices) ;
fprintf (f, '%s\n', 'reals') ;
if isreal(A)
    fprintf (f, '%30.16e\n', real(values)) ;
else
    for k = 1 : nz
	fprintf (f, '%30.16e\n', real(values(k))) ;
	fprintf (f, '%30.16e\n', imag(values(k))) ;
    end	
end

fprintf (f, '%s\n', 'rhs') ;
for j = 1:nrhs
    fprintf (f, '%30.16e\n', b (:,j)) ;
end

fclose (f);

end
