function x = kludemo (A, b)
% x = kludemo (A, b)
%
% Writes out the matrix A as a file, calls the
% stand-alone kludemo program, and reads the
% solution back in.  If b is not provided, it
% is constructed as b = A*xtrue where
% xtrue = 1 + (0:n-1) / n.

[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

% create the files for the matrix A
[Ai j Ax] = find (A) ;
Ap = cumsum ([1 (sum (spones (A)))]) ;
Ai = Ai - 1 ;
Ap = Ap - 1 ;

f = fopen ('tmp/Asize', 'w') ;
fprintf (f, '%d %d %d\n', n, n, nnz (A)) ;
fclose (f) ;

f = fopen ('tmp/Ap', 'w') ;
fprintf (f, '%d\n', Ap) ;
fclose (f) ;

f = fopen ('tmp/Ai', 'w') ;
fprintf (f, '%d\n', Ai) ;
fclose (f) ;

f = fopen ('tmp/Ax', 'w') ;
fprintf (f, '%24.16e\n', Ax) ;
fclose (f) ;

if (nargin > 1)
    f = fopen ('tmp/b', 'w') ;
    fprintf (f, '%24.16e\n', b) ;
    fclose (f) ;
else
    if (exist ('tmp/b', 'file'))
	delete ('tmp/b') ;
    end
end

% solve Ax=b 
! ./kludemo > o

% get the solution
x = load ('tmp/x') ;

