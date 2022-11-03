function test16
%TEST16 test cholmod2 on a large matrix
% Example:
%   test16
% See also cholmod_test

% Copyright 2006-2022, Timothy A. Davis, All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

fprintf ('=================================================================\n');
fprintf ('test16: test cholmod2 on a large matrix\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = ssget (936)							    %#ok
A = Prob.A ;
% tic
% [L,s,p] = lchol (A) ;
% toc
% norm (L,1)

n = size (A,1) ;
b = rand (n,1) ;
tic
x = cholmod2(A,b) ;
t = toc ;
fprintf ('time %g\n', t) ;
err = norm (A*x-b) ;

if (err > 1e-5)
    error ('!') ;
end

fprintf ('test16 passed\n') ;
