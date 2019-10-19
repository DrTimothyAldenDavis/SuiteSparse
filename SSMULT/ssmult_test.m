function ssmult_test
%SSMULT_TEST tests ssmult, ssmultsym (sparse times sparse matrix multiply)
%
% Example:
%   ssmult_test
%
% See also ssmult, ssmult_unsorted, ssmultsym, sstest, sstest2.

% Copyright 2007, Timothy A. Davis, University of Florida

fprintf ('Please wait while your new ssmult function is tested ...\n') ;

ssmult_demo ;

fprintf ('\ntesting large matrices (may fail if you are low on memory):\n') 
rand ('state', 0) ;

n = 10000 ;
A = sprand (n, n, 0.01) ;
B = sprand (n, n, 0.001) ;
test_large (A,B) ;

msg = { 'real', 'complex' } ;

% all of these calls to ssmult should fail:
fprintf ('\ntesting error handling (the errors below are expected):\n') ;
A = { 3, 'gunk', sparse(1), sparse(1), sparse(rand(3,2)) } ;
B = { 4,   0   , 5,         msg,       sparse(rand(3,4)) } ;
for k = 1:length(A)
    try
	% the following statement is supposed to fail 
	C = ssmult (A {k}, B {k}) ;					    %#ok
	error ('test failed\n') ;
    catch
	disp (lasterr) ;
    end
end
fprintf ('error handling tests: ok.\n') ;

% err should be zero:
rand ('state', 0)
for Acomplex = 0:1
    for Bcomplex = 0:1
	err = 0 ;
	fprintf ('\ntesting C = A*B where A is %s, B is %s\n', ...
	    msg {Acomplex+1}, msg {Bcomplex+1}) ;
	for m = [ 0:30 100 ]
	    fprintf ('.') ;
	    for n = [ 0:30 100 ]
		for k = [ 0:30 100 ]
		    A = sprand (m,k,0.1) ;
		    if (Acomplex)
			A = A + 1i*sprand (A) ;
		    end
		    B = sprand (k,n,0.1) ;
		    if (Bcomplex)
			B = B + 1i*sprand (B) ;
		    end
		    C = A*B ;
		    D = ssmult (A,B) ;
		    s = ssmultsym (A,B) ;
		    err = max (err, norm (C-D,1)) ;
		    err = max (err, nnz (C-D)) ;
		    err = max (err, isreal (D) ~= (norm (imag (D), 1) == 0)) ;
		    err = max (err, s.nz > nnz (C)) ;
		    [i j x] = find (D) ;				    %#ok
		    if (~isempty (x))
			err = max (err, any (x == 0)) ;
		    end
		end
	    end
	end
	fprintf (' maximum error: %g\n', err) ;
    end
end

sstest ;
fprintf ('\nSSMULT tests complete.\n') ;



%-------------------------------------------------------------------------------
function test_large (A,B)
% test large matrices
n = size (A,1) ;
fprintf ('dimension %d   nnz(A): %d   nnz(B): %d\n', n, nnz (A), nnz (B)) ;
c = ssmultsym (A,B) ;
fprintf ('nnz(C): %d   flops: %g   memory: %g MB\n', ...
    c.nz, c.flops, c.memory/2^20) ;
try
    % warmup for accurate timings
    C = A*B ;								    %#ok
    D = ssmult (A,B) ;							    %#ok
    E = ssmult_unsorted (A,B) ;						    %#ok
    tic ;
    C = A*B ;
    t1 = toc ;
    tic ;
    D = ssmult (A,B) ;
    t2 = toc ;
    tic ;
    E = ssmult_unsorted (A,B) ;
    t3 = toc ;
    E = E'' ;			    % sort E, to be safe ...
    fprintf ('MATLAB time:          %g\n', t1) ;
    err = norm (C-D,1) ;
    fprintf ('SSMULT time:          %g err: %g\n', t2, err) ;
    err = norm (C-E,1) ;
    fprintf ('SSMULT_UNSORTED time: %g err: %g\n', t3, err) ;
catch
    disp (lasterr)
    fprintf ('tests with large random matrices failed ...\n') ;
end
clear C D E

