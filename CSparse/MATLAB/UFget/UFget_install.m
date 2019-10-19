function UFget_install
%UFget_install installs UFget for the current MATLAB session.
%   Run this M-file (UFget_install) for instructions on how to install it
%   permanently, for future MATLAB sessions.
%
%   See also UFget, PATHTOOL, JAVAADDPATH, ADDPATH.
%   Also see DOC STARTUP.

%   Copyright 2005, Tim Davis, University of Florida.

s = which (mfilename) ;
i = find (s == filesep) ;
s = s (1:i(end)) ;

fprintf ('Temporarily adding %s to your MATLAB path and JAVA path.\n', s) ;
fprintf ('Do this permanently via pathtool.  Next, edit the file:\n') ;
fprintf ('%s file.\n', which ('classpath.txt')) ;
fprintf ('and add the line:\n') ;
fprintf ('%s\n', s) ;
fprintf ('to the end of that file (which defines your JAVA class path).\n') ;

addpath (s) ;
javaaddpath (s) ;

fprintf ('\nAlternatively, add these two lines to your startup.m file:\n\n') ;
fprintf ('addpath (''%s'') ;\n', s) ;
fprintf ('javaaddpath (''%s'') ;\n', s) ;
fprintf ('\nSee also pathtool, javaaddpath, addpath, and "doc startup".\n\n') ;
