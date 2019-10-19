%RBinstall: install the RBio toolbox for use in MATLAB
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
% Copyright 2006, Timothy A. Davis

help RBio

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

mex RBread.f RBrread.f RBcread.f RBread_mex.f RBcsplit.f RBint.f
mex RBtype.f RBwrite.f RBint.f
mex RBwrite.f RBwrite_mex.f RBint.f
mex RBraw.f RBread.f RBint.f

s = pwd ;
addpath (s) ;

t = find (s == filesep) ;
pa = s (1:t(end)) ;
addpath (pa) ;

cd Test
testRB1
cd (s)

fprintf ('\nRBio is ready to use.  Your path has been modified for this\n') ;
fprintf ('session, by adding the following paths:\n') ;
fprintf ('%s\n', s) ;
fprintf ('%s\n', pa) ;
fprintf ('Use the pathtool to modify your path permanently.\n') ;

