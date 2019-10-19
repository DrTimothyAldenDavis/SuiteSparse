function ssmult_install (dotests)
%SSMULT_INSTALL compiles, installs, and tests ssmult.
% Note that the "lcc" compiler provided with MATLAB for Windows can generate
% slow code; use another compiler if possible.  Your current directory must be
% SSMULT for ssmult_install to work properly.  If you use Linux/Unix/Mac, I
% recommend that you use COPTIMFLAGS='-O3 -DNDEBUG' in your mexopts.sh file.
%
% Example:
%   ssmult_install          % compile and install
%   ssmult_install (0)      % just compile and install, do not test
%
% See also ssmult, ssmultsym, ssmult_unsorted, sstest, sstest2, mtimes.

% Copyright 2007, Timothy A. Davis, University of Florida

%-------------------------------------------------------------------------------
% print an introduction
%-------------------------------------------------------------------------------

help ssmult
help ssmult_install

%-------------------------------------------------------------------------------
% compile ssmult and add it to the path
%-------------------------------------------------------------------------------

ssmult_make

addpath (pwd) ;
fprintf ('\nssmult has been compiled, and the following directory has been\n') ;
fprintf ('added to your MATLAB path.  Use pathtool to add it permanently:\n') ;
fprintf ('\n%s\n\n', pwd) ;
fprintf ('If you cannot save your path with pathtool, add the following\n') ;
fprintf ('to your MATLAB startup.m file (type "doc startup" for help):\n') ;
fprintf ('\naddpath (''%s'') ;\n\n', pwd) ;

%-------------------------------------------------------------------------------
% test ssmult and ssmultsym
%-------------------------------------------------------------------------------

if (nargin < 1)
    dotests = 1 ;
end
if (dotests)
    ssmult_test ;
end
