function ssmult_install (dotests)
%SSMULT_INSTALL compiles, installs, and tests ssmult.
% Note that the "lcc" compiler provided with MATLAB for Windows can generate
% slow code; use another compiler if possible.  Your current directory must be
% SSMULT for ssmult_install to work properly.  If you use Linux/Unix/Mac, I
% recommend that you use COPTIMFLAGS='-O3 -DNDEBUG' in your mexopts.sh file.
%
% Example:
%   ssmult_install          % compile and install
%   ssmult_install (0)      % just compile and install, do not test
%
% See also ssmult, ssmultsym, sstest, sstest2, mtimes.

% Copyright 2009, Timothy A. Davis, University of Florida

fprintf ('Compiling SSMULT:\n') ;

%-------------------------------------------------------------------------------
% compile ssmult and add it to the path
%-------------------------------------------------------------------------------

d = '' ;
if (~isempty (strfind (computer, '64')))
    % 64-bit MATLAB
    d = ' -largeArrayDims -DIS64' ;
end

v = getversion ;
if (v < 6.5)
    % mxIsDouble is false for a double sparse matrix in MATLAB 6.1 or earlier
    d = [d ' -DMATLAB_6p1_OR_EARLIER'] ;
end

cmd = sprintf ('mex -O%s ssmult.c ssmult_mex.c ssmult_saxpy.c ssmult_dot.c ssmult_transpose.c', d) ;
disp (cmd) ;
eval (cmd) ;

cmd = sprintf ('mex -O%s ssmultsym.c', d) ;
disp (cmd) ;
eval (cmd) ;

cmd = sprintf ('mex -O%s sptranspose.c ssmult_transpose.c', d) ;
disp (cmd) ;
eval (cmd) ;

addpath (pwd) ;
fprintf ('\nssmult has been compiled, and the following directory has been\n') ;
fprintf ('added to your MATLAB path.  Use pathtool to add it permanently:\n') ;
fprintf ('\n%s\n\n', pwd) ;
fprintf ('If you cannot save your path with pathtool, add the following\n') ;
fprintf ('to your MATLAB startup.m file (type "doc startup" for help):\n') ;
fprintf ('\naddpath (''%s'') ;\n\n', pwd) ;

%-------------------------------------------------------------------------------
% test ssmult and ssmultsym
%-------------------------------------------------------------------------------

if (nargin < 1)
    dotests = 1 ;
end
if (~dotests)
    return
end

fprintf ('Please wait while your new ssmult function is tested ...\n') ;

fprintf ('\nTesting large sparse column vectors (1e7-by-1)\n') ;
x = sprandn (1e7,1,1e-4) ;
y = sprandn (1e7,1,1e-4) ;
x (1) = pi ;
y (1) = exp (1) ;
tic ; a = x'*y ; t1 = toc ;
tic ; b = ssmult (x, y, 1) ; t2 = toc ;
fprintf ('s=x''*y in MATLAB: %8.3f seconds\n', t1) ;
fprintf ('s=ssmult(x,y,1):  %8.3f seconds; error %g\n', t2, abs (full(a-b))) ;
fprintf ('SSMULT speedup: %8.3g\n\n', t1/t2) ;

load west0479
A = west0479 ;
B = sprand (A) ;
C = A*B ;
D = ssmult (A,B) ;
err = norm (C-D,1) / norm (C,1) ;
fprintf ('west0479 error: %g\n', err) ;

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
        C = ssmult (A {k}, B {k}) ;                                         %#ok
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
                    [i j x] = find (D) ;                                    %#ok
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
function [v,pc] = getversion
% determine the MATLAB version, and return it as a double.
% only the primary and secondary version numbers are kept.
% MATLAB 7.0.4 becomes 7.0, version 6.5.2 becomes 6.5, etc.
v = version ;
t = find (v == '.') ;
if (length (t) > 1)
    v = v (1:(t(2)-1)) ;
end
v = str2double (v) ;
try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
end


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
    C = A*B ;                                                               %#ok
    D = ssmult (A,B) ;                                                      %#ok
    tic ;
    C = A*B ;
    t1 = toc ;
    tic ;
    D = ssmult (A,B) ;
    t2 = toc ;
    tic ;
    t3 = toc ;
    fprintf ('MATLAB time:          %g\n', t1) ;
    err = norm (C-D,1) ;
    fprintf ('SSMULT time:          %g err: %g\n', t2, err) ;
catch
    disp (lasterr)
    fprintf ('tests with large random matrices failed ...\n') ;
end
clear C D
