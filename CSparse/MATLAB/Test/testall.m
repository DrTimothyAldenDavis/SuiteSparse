function testall
%TESTALL test all CSparse functions (run tests 1 to 28 below)
%
% Example:
%   testall
% See also: cs_demo

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

clear all
clear functions
cs_test_make	    % compile all CSparse, Demo, Text, and Test mexFunctions

fprintf ('\n------------------------ test1  \n') ; test1  ;
fprintf ('\n------------------------ test2  \n') ; test2  ;
fprintf ('\n------------------------ test3  \n') ; test3  ;
fprintf ('\n------------------------ test4  \n') ; test4  ;
fprintf ('\n------------------------ test5  \n') ; test5  ;
fprintf ('\n------------------------ test6  \n') ; test6  ;
fprintf ('\n------------------------ test7  \n') ; test7  ;
fprintf ('\n------------------------ test8  \n') ; test8  ;
fprintf ('\n------------------------ test9  \n') ; test9  ;
fprintf ('\n------------------------ test10 \n') ; test10 ;
fprintf ('\n------------------------ test11 \n') ; test11 ;
fprintf ('\n------------------------ test12 \n') ; test12 ;
fprintf ('\n------------------------ test13 \n') ; test13 ;
fprintf ('\n------------------------ test14 \n') ; test14 ;
fprintf ('\n------------------------ test15 \n') ; test15 ;
fprintf ('\n------------------------ test16 \n') ; test16 ;
fprintf ('\n------------------------ test17 \n') ; test17 ;
fprintf ('\n------------------------ test18 \n') ; test18 ;
fprintf ('\n------------------------ test19 \n') ; test19 ;
fprintf ('\n------------------------ test20 \n') ; test20 ;
fprintf ('\n------------------------ test21 \n') ; test21 ;
fprintf ('\n------------------------ test22 \n') ; test22 ;
fprintf ('\n------------------------ test23 \n') ; test23 ;
fprintf ('\n------------------------ test24 \n') ; test24 ;
fprintf ('\n------------------------ test25 \n') ; test25 ;
fprintf ('\n------------------------ test26 \n') ; test26 ;
fprintf ('\n------------------------ test27 \n') ; test27 ;
fprintf ('\n------------------------ test28 \n') ; test28 ;
