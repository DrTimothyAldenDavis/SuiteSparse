
function testopts (A,b)

f = fopen ('Afile', 'w') ;

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
fprintf (options, '%s\n', 'default') ;
fclose (options) ;
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
execute (A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
opts (6) = 1 ; %colamd
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 1 ; % btf
opts (6) = 0 ; % amd
opts (7) = 1 ; %scale w.r.t sum
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (8) = 1 ; %continue on singularity
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
opts (6) = 0 ; %amd
opts (7) = 1 ; %scale w.r.t sum
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute(A,b) ;

end

function execute (A, b, opts)
    !klu 

    load x ;

    if (~isreal(A))
	load xz ;
	x = complex(x,xz) ;
    end

    if isreal(A)
	[x2 info] = klus(A,b,opts, []) ;
    else
	[x2 info] = klusz(A,b,opts, []) ;
    end
    if (norm(A*x - b) > 1e-10)
	fprintf('residue : %10.6e  matlab residue : %10.6e\n',...
	         norm(A*x - b),norm(A*x2-b)) ; 
    end
    clear x, x2, xz ;
    load xt ;

    if (~isreal(A))
	load xtz ;
	xt = complex(xt,xtz) ;
    end

    [x2t info] = klust(A, b, opts, []) ;
    if (norm(A'*xt - b) > 1e-10)
	fprintf('transpose residue : %10.6e  matlab residue : %10.6e\n',...
	         norm(A'*xt - b),norm(A'*x2t-b)) ; 
    end
    

    cond2 = mycondest(A) ;
    load cnum ;
    if ((cnum - cond2)/cond2 > 1e-16)
	fprintf('cond : %10.6e matlab cond : %10.6e ratio : %8.3e\n', ...
	        cnum, cond2, cnum/cond2) ;
    end

    load pgrowth ;
    pgrowth2 = pivgrowth(A);
    if ((pgrowth - pgrowth2)/pgrowth2 > 1e-16)
	fprintf('pgrowth: %10.6e matlab pgrowth : %10.6e ratio : %8.3e\n',...
	        pgrowth, pgrowth2, pgrowth/pgrowth2) ;
    end

end
