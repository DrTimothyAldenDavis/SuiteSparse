function [objfiles, timestamp] = cs_make (f)
%CS_MAKE compiles CSparse for use in MATLAB.
%   Usage:
%	cs_make
%	[objfiles, timestamp] = cs_make (f)
%
%   With no input arguments, or with f=0, only those files needing to be
%   compiled are compiled (like the Unix/Linux/GNU "make" command, but not
%   requiring "make").  If f is a nonzero number, all files are compiled.
%   If f is a string, only that mexFunction is compiled.  For example,
%   cs_make ('cs_add') just compiles the cs_add mexFunction.  This option is
%   useful when developing a single new mexFunction.  This function can only be
%   used if the current directory is CSparse/MATLAB/CSparse.  Returns a list of
%   the object files in CSparse, and the latest modification time of any source
%   codes.
%
%   To add a new function and its MATLAB mexFunction to CSparse:
%
%	(1) Create a source code file CSparse/Source/cs_mynewfunc.c.
%	(2) Create a help file, CSparse/MATLAB/CSparse/cs_mynewfunc.m.
%	    This is very useful, but not strictly required.
%	(3) Add the prototype of cs_mynewfunc to CSparse/Source/cs.h.
%	(4) Create its MATLAB mexFunction, CSparse/MATLAB/cs_mynewfunc_mex.c.
%	(5) Edit cs_make.m, and add 'cs_mynewfunc' to the 'cs' and 'csm' lists.
%	(6) Type 'cs_make' in the CSparse/MATLAB/CSparse directory.
%	    If all goes well, your new function is ready for use in MATLAB.
%
%	(7) Optionally add 'cs_mynewfunc' to CSparse/Source/Makefile
%	    and CSparse/MATLAB/CSparse/Makefile, if you want to use the
%	    Unix/Linux/GNU make command instead of cs_make.m.  See where
%	    'cs_add' and 'cs_add_mex' appear in those files, and add
%	    'cs_mynewfunc' accordingly.
%	(8) Optionally add 'cs_mynewfunc' to Tcov/Makefile, and add additional
%	    test code to cs_test.c, and add MATLAB test code to MATLAB/Test/*.
%
%   See also MEX, and the Unix/Linux/GNU MAKE command.

%   Copyright 2006, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

% CSparse source files, in ../../Source, such as ../../Source/cs_add.c.
% Note that not all CSparse source files have their own mexFunction.
cs = { 'cs_add', 'cs_amd', 'cs_chol', 'cs_cholsol', 'cs_counts', ...
    'cs_cumsum', 'cs_dfs', 'cs_dmperm', 'cs_droptol', 'cs_dropzeros', ...
    'cs_dupl', 'cs_entry', 'cs_etree', 'cs_fkeep', 'cs_gaxpy', 'cs_happly', ...
    'cs_house', 'cs_ipvec', 'cs_load', 'cs_lsolve', 'cs_ltsolve', 'cs_lu', ...
    'cs_lusol', 'cs_malloc', 'cs_maxtrans', 'cs_multiply', 'cs_norm', ...
    'cs_permute', 'cs_pinv', 'cs_post', 'cs_print', 'cs_pvec', 'cs_qr', ...
    'cs_qrsol', 'cs_scatter', 'cs_scc', 'cs_schol', 'cs_sqr', 'cs_symperm', ...
    'cs_tdfs', 'cs_transpose', 'cs_compress', 'cs_updown', 'cs_usolve', ...
    'cs_utsolve', 'cs_util', 'cs_reach', 'cs_spsolve', 'cs_ereach', ...
    'cs_leaf', 'cs_randperm' } ;
    % add cs_mynewfunc to the above list

csm = { } ;
if (nargin == 0)
    force = 0 ;
elseif (ischar (f))
    fprintf ('cs_make: compiling ../../Source files and %s_mex.c\n', f) ;
    force = 0 ;
    csm = {f} ;
else
    force = f ;
    if (force)
	fprintf ('cs_make: re-compiling everything\n') ;
    end
end

if (isempty (csm))
    % mexFunctions, of the form cs_add_mex.c, etc, in this directory
    csm = { 'cs_add', 'cs_amd', 'cs_chol', 'cs_cholsol', 'cs_counts', ...
	'cs_dmperm', 'cs_droptol', 'cs_etree', 'cs_gaxpy', 'cs_lsolve', ...
	'cs_ltsolve', 'cs_lu', 'cs_lusol', 'cs_multiply', 'cs_permute', ...
	'cs_print', 'cs_qr', 'cs_qrsol', 'cs_scc', 'cs_symperm', 'cs_thumb', ...
	'cs_transpose', 'cs_sparse', 'cs_updown', 'cs_usolve', ...
	'cs_utsolve', 'cs_randperm' } ;
	% add cs_mynewfunc to the above list
end

if (ispc)
    obj = '.obj' ;
else
    obj = '.o' ;
end

srcdir = sprintf ('..%s..%sSource%s', filesep, filesep, filesep) ;
hfile = [srcdir 'cs.h'] ;

% compile each CSparse source file
[anysrc timestamp] = compile_source ('', 'cs_mex', obj, hfile, force) ;
CS = ['cs_mex' obj] ;
if (nargout > 0)
    objfiles = ['..' filesep 'CSparse' filesep 'cs_mex' obj] ;
end
for i = 1:length (cs)
    [s t] = compile_source (srcdir, cs {i}, obj, hfile, force) ;
    timestamp = max (timestamp, t) ;
    anysrc = anysrc || s ;
    CS = [CS ' ' cs{i} obj] ;
    if (nargout > 0)
	objfiles = [objfiles ' ..' filesep 'CSparse' filesep cs{i} obj] ;
    end
end

% compile each CSparse mexFunction
obj = ['.' mexext] ;
for i = 1:length (csm)
    [s t] = cs_must_compile ('', csm{i}, '_mex', obj, hfile, force) ;
    timestamp = max (timestamp, t) ;
    if (anysrc || s)
	cmd = sprintf ('mex -O -I../../Source %s_mex.c %s -output %s\n', ...
	    csm{i}, CS, csm{i}) ;
	fprintf ('%s', cmd) ;
	eval (cmd) ;
    end
end

end

%-------------------------------------------------------------------------------
function [s,t] = compile_source (srcdir, f, obj, hfile, force)
% compile a source code file in ../../Source, leaving object file in
% this directory.
[s t] = cs_must_compile (srcdir, f, '', obj, hfile, force) ;
if (s)
    cmd = sprintf ('mex -O -c -I../../Source %s%s.c\n', srcdir, f) ;
    fprintf ('%s', cmd) ;
    eval (cmd) ;
end
end
