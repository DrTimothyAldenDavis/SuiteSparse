function utest (A,b)
% UTEST: test KLU
% Example:
%   utest (A,b)

[m n] = size (A) ;

b = kwrite (A,b) ;

execute_test (A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
opts (6) = 1 ; %colamd
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 1 ; % btf
opts (6) = 0 ; % amd
opts (7) = 1 ; %scale w.r.t sum
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (8) = 1 ; %continue on singularity
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (5) = 0 ; %no btf
opts (6) = 0 ; %amd
opts (7) = 1 ; %scale w.r.t sum
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test(A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test (A,b) ;

options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (6) = 2 ; % user generated ordering
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test (A,b) ;

%multiple rhs : 2
opts (6) = 0 ; %amd
b = rand (m, 2) ;
[mb nrhs] = size (b) ;
kwrite (A,b) ;
execute_test (A,b, nrhs) ;

opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;
options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test (A,b, nrhs) ;

%multiple rhs : 3
b = rand (m, 3) ;
[mb nrhs] = size (b) ;
kwrite (A,b) ;
execute_test (A,b, nrhs) ;

opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;
options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test (A,b, nrhs) ;

%multiple rhs : 4
b = rand (m, 4) ;
[mb nrhs] = size (b) ;
kwrite (A,b) ;
execute_test (A,b, nrhs) ;

opts = [0.001 1.2 1.2 10 1 0 0 0 ] ;
options = fopen ('options', 'w') ;
fprintf (options, '%s\n', 'options') ;
% use different options
opts (7) = 2 ; %scale w.r.t max
fprintf (options, '%g\n', opts) ;
fclose(options) ;
execute_test (A,b, nrhs) ;
%!coverage ;
end

% ------------------------------------------------------------------------------

function execute_test (A, b, nrhs)

    % run the ut program to test KLU on the matrix
    % eval ('!./ut') ;
    eval ('!valgrind -q ut') ;

    load x ;

    if (~isreal(A))
	x = 0 ;
	xz = 0 ;
	load xz ;
	x = complex(x,xz) ;
	clear xz
    end
%    size(x)
    [m,n] = size(A) ;
    if (nargin < 3)
	nrhs = 1 ;
    end

    for j = 1:nrhs
        x2 = A \ b(:,j) ;
	begin = (j - 1) * m + 1 ;
        endl =  j * m ;
	if (norm( x(begin:endl) - x2) > 1e-16)
	    fprintf('          resid : %10.6e  matlab resid : %10.6e\n',...
	    norm(A*x(begin:endl) - b(:,j)),norm(A*x2-b(:,j))) ; 
	end
    end
    clear x x2
    xt = 0 ;
    xtz = 0 ;
    load xt

    if (~isreal(A))
	load xtz
	xt = complex(xt,xtz) ;
    end

    for j = 1:nrhs
        x2t = A' \ b(:,j) ;
	begin = (j - 1) * m + 1 ;
        endl =  j * m ;
	if (norm( xt(begin:endl) - x2t) > 1e-16)			    %#ok
	    fprintf('transpose resid : %10.6e  matlab resid : %10.6e\n',...
	    norm(A'*xt(begin:endl) - b(:,j)),norm(A'*x2t-b(:,j))) ; 
        end
    end

    if (0)
	cond2 = mycondest(A) ;
	load cnum ;
	if (abs(cnum - cond2)/cond2 > 1e-16)
	    fprintf('cond : %10.6e matlab cond : %10.6e ratio : %8.3e\n', ...
		    cnum, cond2, cnum/cond2) ;
	end

	load pgrowth ;
	pgrowth2 = pivgrowth(A);
	if (abs(pgrowth - pgrowth2)/pgrowth2 > 1e-16)
	   fprintf('pgrowth: %10.6e matlab pgrowth : %10.6e ratio : %8.3e\n',...
		    pgrowth, pgrowth2, pgrowth/pgrowth2) ;
	end
    end

end
