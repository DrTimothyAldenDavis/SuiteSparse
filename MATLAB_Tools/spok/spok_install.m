function spok_install
%SPOK_INSTALL compiles and installs the SPOK mexFunction
% Your current working directory must be the "spok" directory for this function
% to work.
%
% Example:
%   spok_install
%
% See also sparse, spok, spok_test

% Copyright 2008-2011, Tim Davis, University of Florida

help spok_install
is64 = ~isempty (strfind (computer, '64')) ;
if (is64)
    mex -largeArrayDims  spok.c spok_mex.c
else
    mex spok.c spok_mex.c
end
addpath (pwd)

fprintf ('Added the following directory to your path:\n') ;
disp (pwd) ;
fprintf ('Use pathtool and click "save" to save this for future sessions.\n') ;
