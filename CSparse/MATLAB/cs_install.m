function cs_install (do_pause)
%CS_INSTALL: compile and install CSparse for use in MATLAB.
%   Your current working directory must be CSparse/MATLAB in order to use this
%   function.
%   
%   The directories
%
%       CSparse/MATLAB/CSparse
%       CSparse/MATLAB/Demo
%       CSparse/MATLAB/UFget
%
%   are added to your MATLAB path (see the "pathtool" command to add these to
%   your path permanently, for future MATLAB sessions).  The path
%
%       CSparse/MATLAB/UFget
%
%   is also added to your java class path (see the "javaaddpath" command).
%   Edit your classpath.txt file (type the command "which classpath.txt") to
%   add this directory to your Java class path permanently.  UFget is not
%   required by CSparse (except the extended demos and tests); UFget requires
%   MATLAB 7.0 or later.
%
%   Next, the MATLAB CSparse demo program, CSparse/MATLAB/cs_demo is executed.
%   To run the demo with pauses so you can see the results, use cs_install(1).
%   To run the full MATLAB test programs for CSparse, run testall in the
%   Test directory.
%
%   Example:
%       cs_install          % install and run demo with no pauses
%       cs_install(1)       % install and run demo with pauses
%
%   See also: cs_demo
%
%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

if (nargin < 1)
    do_pause = 0 ;
end

help cs_install

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

if (do_pause)
    input ('Hit enter to continue: ') ;
end
addpath ([pwd filesep 'CSparse']) ;
addpath ([pwd filesep 'Demo']) ;
addpath ([pwd filesep 'UFget']) ;

ufget_path = [pwd filesep 'UFget'] ;
try
    javaaddpath (ufget_path) ;
catch
    fprintf ('It appears that you are using an older version of MATLAB.\n') ;
    fprintf ('To use UFget, edit your classpath.txt file and add the line:\n');
    fprintf ('%s\n', ufget_path) ;
    fprintf ('Note that UFget requires MATLAB 7.0 or later.\n') ;
    fprintf ('The location of classpath.txt is given below:\n') ;
    which classpath.txt
end

cd ('CSparse') ;
cs_make
cd ('../Demo') ;
cs_demo (do_pause)
