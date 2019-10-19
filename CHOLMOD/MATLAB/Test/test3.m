function test3
%TEST3 test sparse on int8, int16, and logical
% Example:
%   test3
% See also cholmod_test

% Copyright 2006-2007, Timothy A. Davis, University of Florida

fprintf ('=================================================================\n');
fprintf ('test3: test sparse on int8, int16, and logical\n') ;

clear all
c =  ['a' 'b' 0 'd']							    %#ok
sparse(c)
sparse2(c)
sparse(c')
sparse2(c')
whos
nzmax(ans)  %#ok

try % this will fail
    sparse(int8(c))
catch
    fprintf ('sparse(int8(c)) fails in MATLAB\n') ;
end
sparse2(int8(c))

sparse2 (int16(c))
whos
s = logical(rand(4) > .5)						    %#ok
sparse (s)
whos
sparse2(s)
whos

x = rand(4)								    %#ok
sparse (x > .5)								    %#ok
whos
sparse2 (x > .5)							    %#ok
whos

fprintf ('test3 passed\n') ;
