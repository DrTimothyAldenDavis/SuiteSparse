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
%   your path permanently, for future MATLAB sessions).
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
%   Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('Compiling and installing CSparse\n') ;
if (nargin < 1)
    do_pause = 0 ;
end

if (do_pause)
    input ('Hit enter to continue: ') ;
end
addpath ([pwd '/CSparse']) ;
addpath ([pwd '/Demo']) ;

if (verLessThan ('matlab', '7.0'))
    fprintf ('UFget not installed (MATLAB 7.0 or later required)\n') ;
else
    % install UFget, unless it's already in the path
    try
        % if this fails, then UFget is not yet installed
        index = UFget ;
        fprintf ('UFget already installed:\n') ;
        which UFget
    catch
        index = [ ] ;
    end
    if (isempty (index))
        % UFget is not installed.  Use ./UFget
        fprintf ('Installing ./UFget\n') ;
        try
            addpath ([pwd '/UFget']) ;
        catch me
            disp (me.message) ;
            fprintf ('UFget not installed\n') ;
        end
    end
end

cd ('CSparse') ;
cs_make (1) ;
cd ('../Demo') ;
cs_demo (do_pause)
