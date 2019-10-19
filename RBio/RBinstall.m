%RBINSTALL install the RBio toolbox for use in MATLAB
% Compiles the Fortran mexFunctions RBread, RBwrite, RBtype, and RBraw, and
% the C mexFunction UFfull_write, and adds the current directory to the MATLAB
% path.
%
% Example:
%
%   RBinstall
%
% See also RBread, RBwrite, RBtype, RBraw.
%
% Copyright 2007, Timothy A. Davis

help RBio

RBmake

s = pwd ;
addpath (s) ;

cd Test
testRB1
cd (s)

fprintf ('\nRBio is ready to use.  Your path has been modified for this\n') ;
fprintf ('session, by adding the following path:\n') ;
fprintf ('%s\n', s) ;
fprintf ('Use the pathtool to modify your path permanently.\n') ;

