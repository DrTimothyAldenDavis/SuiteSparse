function testopts (A,b)
% testopts: test KLU
% Example:
%   testops(A,b)

f = fopen ('Afile', 'w') ;

[m n] = size (A) ;
[mb nrhs] = size (b) ;

opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;

if (m ~= n)
    error ('A must be square') ;
end
if (mb ~= m)
    error ('A and b must have same # rows') ;
end

if (nargin < 2)
    b = kwrite (A) ;
else
    kwrite (A,b) ;
end
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

    x = 0 ;
    xz = 0 ;
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
    clear x x2 xz
    xt = 0 ;
    xtz = 0 ;
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
