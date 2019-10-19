function cs_demo (do_pause)
%CS_DEMO run all CSparse demos.
%   Your current working directory must be CSparse/MATLAB/Demo to run this demo.
%   cs_demo(0) will run all demos without pausing.
%
% Example:
%   cs_demo
% See also: cs_demo1, cs_demo2, cs_demo3

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

help cs_demo
if (nargin < 1)
    do_pause = 1 ;
end

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo1
cs_demo1 ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo2
cs_demo2 (do_pause) ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo3
cs_demo3 (do_pause) ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help ex_1
ex_1

fprintf ('\n\n-------------------------------------------------------\n') ;
help ex2
ex2

fprintf ('\n\n-------------------------------------------------------\n') ;
help ex3
ex3

fprintf ('\nAll CSparse demos finished.\n') ;
