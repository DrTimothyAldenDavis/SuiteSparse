function RBinstall (quiet)
%RBINSTALL install the RBio toolbox for use in MATLAB
% Compiles the mexFunctions RBread, RBwrite, RBtype, and RBraw, and adds the
% current directory to the MATLAB path.
%
% Example:
%
%   RBinstall
%
% See also RBread, RBwrite, RBtype, RBraw.
%
% RBio, Copyright (c) 2009-2022, Timothy A. Davis.  All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin < 1)
    quiet = 0 ;
end

if (~quiet)
    help RBio
end

RBmake

s = pwd ;
addpath (s) ;

cd private
testRB1
if (exist ('ssget') == 2) %#ok<EXIST>
    testRB2
end
cd (s)

if (~quiet)
    fprintf ('\nRBio is ready to use.  Your path has been modified for\n') ;
    fprintf ('this session, by adding the following path:\n') ;
    fprintf ('%s\n', s) ;
    fprintf ('Use the pathtool to modify your path permanently.\n') ;
end

