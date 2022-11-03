function testh
%TESTH test Householder reflections
%
% Example:
%   testh
% See also: testall

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

format long e
fprintf ('-------------------------------------------------\n') ;
x = [-3 4 5]' ;
disp (x) ;
[v, beta, s] = house (x) ;
disp ('v = ') ;
disp (v) ;
disp ('beta = ') ;
disp (beta) ;
disp ('s = ') ;
disp (s) ;
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;


x = [3 4 5]' ;
disp (x) ;
[v, beta, s] = house (x) ;
disp ('v = ') ;
disp (v) ;
disp ('beta = ') ;
disp (beta) ;
disp ('s = ') ;
disp (s) ;
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;

x = [1 eps]' ;
disp (x) ;
[v, beta, s] = house (x) ;
disp ('v = ') ;
disp (v) ;
disp ('beta = ') ;
disp (beta) ;
disp ('s = ') ;
disp (s) ;
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;

x = pi ;
disp (x) ;
[v, beta, s] = house (x) ;
disp ('v = ') ;
disp (v) ;
disp ('beta = ') ;
disp (beta) ;
disp ('s = ') ;
disp (s) ;
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;

x = -pi ;
disp (x) ;
[v, beta, s] = house (x) ;
disp ('v = ') ;
disp (v) ;
disp ('beta = ') ;
disp (beta) ;
disp ('s = ') ;
disp (s) ;
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;

x = [1 0 0]' ;
disp (x) ;
[v, beta, s] = house (x) ;                                                  %#ok
x = x - v*(beta*(v'*x)) ;
disp (x) ;
fprintf ('-------------------------------------------------\n') ;
