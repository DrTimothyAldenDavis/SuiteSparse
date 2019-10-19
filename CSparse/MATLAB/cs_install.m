function cs_install
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
%   add this directory to your Java class path permanently.
%
%   Next, the MATLAB CSparse demo program, CSparse/MATLAB/cs_demo is executed.
%   To run the full MATLAB test programs for CSparse, run testall in the
%   Test directory.
%
%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

help cs_install
input ('Hit enter to continue: ') ;
addpath ([pwd filesep 'CSparse']) ;
addpath ([pwd filesep 'Demo']) ;
addpath ([pwd filesep 'UFget']) ;
javaaddpath ([pwd filesep 'UFget']) ;

cd ('CSparse') ;
cs_make
cd ('../Demo') ;
cs_demo
