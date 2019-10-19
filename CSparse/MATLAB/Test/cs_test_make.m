function cs_test_make (force)
%CS_TEST_MAKE compiles the CSparse, Demo, and Test mexFunctions.
%   The current directory must be CSparse/MATLAB/Test to use this function.
%
% Example:
%   cs_test_make
% See also: testall

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

if (nargin < 1)
    force = 0 ;
end

cd ('../CSparse') ;
[object_files timestamp] = cs_make ;
cd ('../Test') ;

mexfunc = { 'cs_ipvec', 'cs_pvec', 'cs_sparse2', ...
    'cs_reach', 'cs_maxtransr', 'cs_reachr', 'cs_rowcnt', 'cs_frand' } ;

for i = 1:length(mexfunc)
    [s t tobj] = cs_must_compile ('', mexfunc{i}, '_mex', ...
        ['.' mexext], 'cs_test_make.m', force) ;
    if (s | tobj < timestamp)                                               %#ok
        cmd = ['mex -O -output ' mexfunc{i} ' ' mexfunc{i} '_mex.c -I..' ...
            filesep '..' filesep 'Include -I..' ...
            filesep 'CSparse ' object_files] ;
        fprintf ('%s\n', cmd) ;
        eval (cmd) ;
     end
end
