function Problem = ssread (directory, tmp)
%SSREAD read a Problem in Matrix Market, Rutherford/Boeing, or Binsparse format
% containing a set of files created by sswrite.  See sswrite for a description
% of the Problem struct.
%
% Usage: Problem = ssread (directory)
%
% Example:
%
%   load west0479
%   clear Problem
%   Problem.name = 'HB/west0479' ;
%   Problem.title = '8 STAGE COLUMN SECTION, ALL SECTIONS RIGOROUS (CHEM.ENG.)';
%   Problem.A = west0479 ;
%   Problem.id = 267 ;          % the id number of west0479 in the collection
%   Problem.date = '1983' ;
%   Problem.author = 'A. Westerberg' ;
%   Problem.ed = 'I. Duff, R. Grimes, J. Lewis'
%   Problem.kind = 'chemical process simulation problem' ;
%   sswrite (Problem, 'RB/', '') ;
%   Prob3 = ssread ('RB/HB/west0479')
%   isequal (Problem, Prob3)
%
% This part of the example requires CHOLMOD, for the mread function:
%
%   sswrite (Problem, 'MM/') ;
%   Prob2 = ssread ('MM/HB/west0479')
%   isequal (Problem, Prob2)
%
% You can also compare this Problem with the version in the SuiteSparse Matrix
% Collection, with ssget(267) or ssget('HB/west0479').  Note that this includes
% the 22 explicit zero entries present in the west0479 Harwell/Boeing matrix,
% but not included in the MATLAB west0479.mat demo matrix.  Those entries are
% present in the SuiteSparse Matrix Collection.  This example assumes your current
% directory is the RBio directory, containing the west0479 problem in the
% RBio/Test directory:
%
%   Prob5 = ssget ('HB/west0479')
%   Prob6 = ssread ('Test/west0479')
%   isequal (Prob5, Prob6)
%
% The directory can be a compressed tar file of the form "name.tar.gz", in
% which case the tarfile is uncompressed into a temporary directory, and
% the temporary directory is deleted when done.  The '.tar.gz' should not be
% part of the directory argument.  In this case, a 2nd input argument can be
% provided:  Problem = ssread (directory, tmp).  The problem is extracted into
% the tmp directory.  If tmp is not present, the output of the tempdir function
% is used instead.
%
% A Binsparse Problem can instead be read from its Name.bsp.h5 file.  Either
% the complete filename or the path without the .bsp.h5 extension is accepted.
%
% Note that ssget is much faster than ssread.  ssread is useful if you are
% short on disk space, and want to have just one copy of the collection that
% can be read by MATLAB (via ssread) and a non-MATLAB program (the MM, RB, or
% Binsparse versions of the collection).
%
% Reading Binsparse output requires binsparse_read,
% binsparse_read_string_dataset, and binsparse_to_ssmc_problem from the
% Binsparse MATLAB bindings.
%
% See also sswrite, mread, mwrite, RBread, ssget, untar, tempdir.

% Optionally uses the CHOLMOD mread mexFunction, for reading Problems in
% Matrix Market format.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

%-------------------------------------------------------------------------------
% determine the Problem name from the directory name
%-------------------------------------------------------------------------------

directory = regexprep (directory, '[\/\\]', '/') ;
t = find (directory == '/') ;
if (isempty (t))
    name = directory ;
else
    name = directory (t(end)+1:end) ;
end

%-------------------------------------------------------------------------------
% read a standalone Binsparse file
%-------------------------------------------------------------------------------

is_bsp_file = (length (directory) >= 7 && ...
    strcmpi (directory (end-6:end), '.bsp.h5')) ;
if (exist (directory, 'file') == 2 && is_bsp_file)
    Problem = read_bsp_problem (directory) ;
    return
end
bspfile = [directory '.bsp.h5'] ;
if (exist (bspfile, 'file') == 2)
    Problem = read_bsp_problem (bspfile) ;
    return
end

%-------------------------------------------------------------------------------
% open the directory, or untar the tar.gz file
%-------------------------------------------------------------------------------

d = dir (directory) ;
is_tar = 0 ;

if (isempty (d))
    % look for a .tar.gz file
    if (nargin < 2)
	tmpdir = [tempname '_ssread_' name] ;
    else
	tmpdir = [tmp '/' name] ;
    end
    try
	% try untaring the problem
	untar ([directory '.tar.gz'], tmpdir) ;
    catch
	% untar failed, make sure tmpdir is deleted
	try
	    rmdir (tmpdir, 's') ;
	catch
	end
	error (['unable to read problem: ' directory]) ;
    end
    directory = [tmpdir '/' name] ;
    d = dir (directory) ;
    is_tar = 1 ;
end 

%-------------------------------------------------------------------------------
% read the problem
%-------------------------------------------------------------------------------

try

    %---------------------------------------------------------------------------
    % get name, title, id, kind, date, author, editor, notes from master file
    %---------------------------------------------------------------------------

    masterfile = [directory '/' name] ;
    bspfile = [masterfile '.bsp.h5'] ;
    if (exist (bspfile, 'file') == 2)
        Problem = read_bsp_problem (bspfile) ;
        if (is_tar)
            rmdir (tmpdir, 's') ;
        end
        return
    end
    [Problem notes RB] = get_header (masterfile) ;

    %---------------------------------------------------------------------------
    % get the A and Zero matrices from the master file and add to the Problem
    %---------------------------------------------------------------------------

    if (RB)
	% read in the primary Rutherford/Boeing file
	[Problem.A Zeros] = RBread ([masterfile '.rb']) ;
    else
	% read in the primary Matrix Market file.  Get patterns as binary.
	[Problem.A Zeros] = mread ([masterfile '.mtx'], 1) ;
    end
    if (nnz (Zeros) > 0)
	Problem.Zeros = Zeros ;
    end

    % add the notes after A and Zeros
    if (~isempty (notes))
	Problem.notes = notes ;
    end

    namelen = length (name) ;

    %---------------------------------------------------------------------------
    % read b, x, aux (incl. any aux.cell sequences), stored as separate files
    %---------------------------------------------------------------------------

    for k = 1:length(d)

	% get the next filename in the directory
	file = d(k).name ;
	fullfilename = [directory '/' file] ;

	if (length (file) < length (name) + 1)

	    % unrecognized file; skip it
	    continue

	elseif (strcmp (file, [name '.mtx']))

	    % skip the master file; already read in
	    continue

	elseif (strcmp (file, [name '_b.mtx']))

	    % read in b as a Matrix Market file
	    Problem.b = mtx_read (fullfilename, RB) ;

	elseif (strcmp (file, [name '_x.mtx']))

	    % read in x as a Matrix Market file
	    Problem.x = mtx_read (fullfilename, RB) ;

	elseif (strcmp (file, [name '_b.rb']))

	    % read in b as a Rutherford/Boeing file
	    Problem.b = RBread (fullfilename) ;

	elseif (strcmp (file, [name '_x.rb']))

	    % read in x as a Rutherford/Boeing file
	    Problem.x = RBread (fullfilename) ;

	elseif (strcmp (file (1:length(name)+1), [name '_']))

	    % read in an aux component, in the form name_whatever.mtx
	    thedot = find (file == '.', 1, 'last') ;
	    ext = file (thedot:end) ;

	    if (strcmp (ext, '.txt'))

                % get a txt file as either a char array or cell array of strings
                C = sstextread (fullfilename, Problem.id > 2776) ;

	    elseif (strcmp (ext, '.mtx'))

		% read a full or sparse auxiliary matrix in the Matrix Market
		% form, or a full auxiliary matrix in the Rutherford/Boeing form.
		C = mtx_read (fullfilename, RB) ;

	    elseif (strcmp (ext, '.rb'))

		% read in a sparse matrix, for a Rutherford/Boeing collection
		C = RBread (fullfilename) ;

	    else

		% this file is not recognized - skip it.
		C = [ ] ;

	    end

	    % determine the name of the component and place it in the Problem
	    if (~isempty (C))
		% Determine if this is part of an aux.whatever cell sequence.
		% These filenames have the form name_whatever_#.mtx, where name
		% is the name of the Problem, and # is a number (1 or more
		% digts) greater than zero.  If # = i, this becomes the
		% aux.whatever{i} matrix.
		suffix = file (namelen+2:thedot-1) ;
		t = find (suffix == '_', 1, 'last') ;
		what = suffix (1:t-1) ;
		i = str2num (suffix (t+1:end)) ;			    %#ok
		if (~isempty (i) && i > 0 && ~isempty (what))
		    % this is part of aux.whatever{i} cell array
		    Problem.aux.(what) {i,1} = C ;
		elseif (~isempty (suffix))
		    % this is not a cell, simply an aux.whatever matrix
		    Problem.aux.(suffix) = C ;
		end
	    end
	end
    end

    %---------------------------------------------------------------------------
    % delete the uncompressed version of the tar file
    %---------------------------------------------------------------------------

    if (is_tar)
	rmdir (tmpdir, 's') ;
    end

catch

    %---------------------------------------------------------------------------
    % catch the error, delete the temp directory, and rethrow the error
    %---------------------------------------------------------------------------

    try
	if (is_tar)
	    rmdir (tmpdir, 's') ;
	end
    catch
    end
    rethrow (lasterror) ;

end


%-------------------------------------------------------------------------------
% get_header: get the header of the master file (Group/name/name.txt or .mtx)
%-------------------------------------------------------------------------------

function [Problem, notes, RB] = get_header (masterfile)
% Get the name, title, id, kind, date, author, editor and notes from the master
% file.  The name, title, and id are required.  They appear as structured
% comments in the Matrix Market file (masterfile.mtx) or in the text file for
% a problem in Rutherford/Boeing format (masterfile.txt).  RB is returned as
% 1 if the problem is in Rutherford/Boeing format, 0 otherwise.

% first assume it's in Matrix Market format
f = fopen ([masterfile '.mtx'], 'r') ;
if (f < 0)
    % oops, that failed.  This must be a problem in Rutherford/Boeing format
    RB = 1 ;
    f = fopen ([masterfile '.txt'], 'r') ;
    if (f < 0)
	% oops again, this is not a valid problem in the SuiteSparse collection
	error (['invalid problem: ' masterfile]) ;
    end
else
    % we found the Matrix Market file
    RB = 0 ;
end
Problem = [ ] ;
notes = [ ] ;

while (1)

    % get the next line
    s = fgetl (f) ;
    if (~ischar (s) || length (s) < 3 || s (1) ~= '%')
	% end of file or end of leading comments ... no notes found
	fclose (f) ;
	[Problem notes] = valid_problem (Problem, [ ]) ;
	return ;
    end

    % remove the leading '% ' and get the first token
    s = s (3:end) ;
    [t r] = strtok (s) ;

    % parse the line
    if (strcmp (t, 'name:'))

	% get the Problem.name.  It must be of the form Group/Name.
	Problem.name = strtrim (r) ;
	if (length (find (Problem.name == '/')) ~= 1)
	    fclose (f) ;
	    error (['invalid problem name ' Problem.name]) ;
	end

    elseif (s (1) == '[')

	% get the Problem.title
	k = find (s == ']', 1, 'last') ;
	if (isempty (k))
	    fclose (f) ;
	    error ('invalid problem title') ;
	end
	Problem.title = s (2:k-1) ;

    elseif (strcmp (t, 'id:'))

	% get the Problem.id
	Problem.id = str2num (r) ;					    %#ok
	if (isempty (Problem.id) || Problem.id < 0)
	    fclose (f) ;
	    error ('invalid problem id') ;
	end

    elseif (strcmp (t, 'kind:'))

	% get the Problem.kind
	Problem.kind = strtrim (r) ;

    elseif (strcmp (t, 'date:'))

	% get the Problem.date
	Problem.date = strtrim (r) ;

    elseif (strcmp (t, 'author:'))

	% get the Problem.author
	Problem.author = strtrim (r) ;

    elseif (strcmp (t, 'ed:'))

	% get the Problem.ed
	Problem.ed = strtrim (r) ;

    elseif (strcmp (t, 'notes:'))

	% get the notes, which always appear last
	k = 0 ;
	notes = [ ] ;
	while (1)
	    % get the next line
	    s = fgetl (f) ;
	    if (~ischar (s) || length (s) < 2 || ~strcmp (s (1:2), '% '))
		% end of file or end of notes ... convert notes to char array
		fclose (f) ;
		[Problem notes] = valid_problem (Problem, notes) ;
		return ;
	    end
	    % add the line to the notes
	    k = k + 1 ;
	    notes {k} = s ;						    %#ok
	end
    end
end


%-------------------------------------------------------------------------------
% valid_problem: determine if a problem is valid, and finalizes the notes
%-------------------------------------------------------------------------------

function [Problem, notes] = valid_problem (Problem, notes)
% make sure the required fields (name, title, id, date, author, ed) are present.
% Convert notes to char, and strip off the leading '% ', inserted when the notes
% were printed in the Matrix Market file.
if (~isfield (Problem, 'name') || ~isfield (Problem, 'title') || ...
    ~isfield (Problem, 'id') || ~isfield (Problem, 'date') || ...
    ~isfield (Problem, 'author') || ~isfield (Problem, 'ed') || ...
    ~isfield (Problem, 'kind'))
    error ('invalid Problem mfile') ;
end
if (~isempty (notes))
    notes = char (notes) ;
    notes = notes (:, 3:end) ;
end


%-------------------------------------------------------------------------------
% mtx_read: read a *.mtx file
%-------------------------------------------------------------------------------

% In the Rutherford/Boeing form, a *.mtx file is used only for full matrices,
% using a tiny subset of the Matrix Market format.  In the Matrix Market form,
% the *.mtx is used for all b, x, and aux matrices (both full and sparse).

function C = mtx_read (file, RB)

if (~RB)

    % Get a Matrix Market file, using full Matrix Market features.
    C = mread (file, 1) ;

else

    % mread is not installed.  The RB format uses a tiny subset of the Matrix
    % Market format for full matrices: just the one header line, and no comment
    % or blank lines permitted.  Allowable header lines are:
    %	%%MatrixMarket matrix array real general
    %	%%MatrixMarket matrix array complex general
    % This tiny subset can be read by ssfull_read.
    C = ssfull_read (file) ;

end


%-------------------------------------------------------------------------------
% read_bsp_problem: read and convert one Binsparse SSMC problem
%-------------------------------------------------------------------------------

function Problem = read_bsp_problem (bspfile)

reader = which ('binsparse_read') ;
[~, ~, reader_extension] = fileparts (reader) ;
if (isempty (reader) || ~strcmpi (reader_extension, ['.' mexext]))
    error ('SuiteSparse:ssread:MissingBinsparseReader', ...
        'BSP input requires the binsparse_read MEX function') ;
end
if (isempty (which ('binsparse_to_ssmc_problem')))
    error ('SuiteSparse:ssread:MissingBinsparseConverter', ...
        'BSP input requires binsparse_to_ssmc_problem') ;
end
if (exist ('binsparse_read_string_dataset', 'file') ~= 3)
    error ('SuiteSparse:ssread:MissingBinsparseStringReader', ...
        'BSP input requires the binsparse_read_string_dataset MEX function') ;
end

try
    descriptor_text = h5readatt (bspfile, '/', 'binsparse') ;
    descriptor = jsondecode (char (descriptor_text)) ;
catch me
    error ('SuiteSparse:ssread:InvalidBinsparseMetadata', ...
        'unable to read BSP metadata: %s', me.message) ;
end
if (~isstruct (descriptor) || ~isfield (descriptor, 'metadata') || ...
        ~isstruct (descriptor.metadata))
    error ('SuiteSparse:ssread:InvalidBinsparseMetadata', ...
        'BSP file does not contain SuiteSparse problem metadata') ;
end

bsp_problem = struct ;
bsp_problem.metadata = descriptor.metadata ;
bsp_problem.A = binsparse_read (bspfile) ;

[groups, datasets] = bsp_root_members (bspfile) ;
for k = 1:numel (groups)
    component = bsp_component_name (groups {k}) ;
    value = binsparse_read (bspfile, component) ;
    bsp_problem = add_bsp_component (bsp_problem, component, value) ;
end

reserved = {'values', 'indices_0', 'indices_1', 'pointers_to_1'} ;
for k = 1:numel (datasets)
    component = datasets {k} ;
    if (any (strcmp (component, reserved)))
        continue
    end
    value = binsparse_read_string_dataset (bspfile, ['/' component]) ;
    bsp_problem = add_bsp_component (bsp_problem, component, value) ;
end

Problem = binsparse_to_ssmc_problem (bsp_problem) ;


%-------------------------------------------------------------------------------
% bsp_root_members: list the groups and datasets in the root of a BSP file
%-------------------------------------------------------------------------------

function [groups, datasets] = bsp_root_members (bspfile)

% h5info is deliberately not used here.  It describes every object in the
% file, and the description of a dataset includes the size of its datatype,
% which h5info builds an array for.  A fixed-length HDF5 string is one such
% datatype, and its size is the whole width of the text, so a single wide
% string is enough to make h5info fail with MATLAB:pmaxsize ("Requested array
% exceeds the maximum possible variable size") no matter how much memory the
% machine has.  The threshold is 2GB, and SuiteSparse problems reach it: the
% aux.names of Sybrandt/AGATHA_2015 is one 3.2GB row of text.  The low-level
% interface below reports only names and object types, so it reads such a
% file without ever sizing the datatype.

groups = { } ;
datasets = { } ;

file = H5F.open (bspfile, 'H5F_ACC_RDONLY', 'H5P_DEFAULT') ;
closefile = onCleanup (@ ( ) H5F.close (file)) ;
root = H5G.open (file, '/') ;
closeroot = onCleanup (@ ( ) H5G.close (root)) ;

is_group = H5ML.get_constant_value ('H5O_TYPE_GROUP') ;
is_dataset = H5ML.get_constant_value ('H5O_TYPE_DATASET') ;

info = H5G.get_info (root) ;
for k = 0:double (info.nlinks) - 1
    name = H5L.get_name_by_idx (root, '.', 'H5_INDEX_NAME', 'H5_ITER_INC', ...
        k, 'H5P_DEFAULT') ;
    object = H5O.open_by_idx (root, '.', 'H5_INDEX_NAME', 'H5_ITER_INC', ...
        k, 'H5P_DEFAULT') ;
    closeobject = onCleanup (@ ( ) H5O.close (object)) ;
    object_info = H5O.get_info (object) ;
    kind = object_info.type ;
    clear closeobject
    if (kind == is_group)
        groups {end+1} = name ;                                     %#ok<AGROW>
    elseif (kind == is_dataset)
        datasets {end+1} = name ;                                   %#ok<AGROW>
    end
end


%-------------------------------------------------------------------------------
% bsp_component_name
%-------------------------------------------------------------------------------

function name = bsp_component_name (path)

slash = find (path == '/', 1, 'last') ;
if (isempty (slash))
    name = path ;
else
    name = path (slash+1:end) ;
end
if (isempty (name) || ~isvarname (name))
    error ('SuiteSparse:ssread:InvalidBinsparseComponent', ...
        'invalid BSP component name: %s', path) ;
end


%-------------------------------------------------------------------------------
% add_bsp_component
%-------------------------------------------------------------------------------

function bsp_problem = add_bsp_component (bsp_problem, name, value)

if (~isvarname (name))
    error ('SuiteSparse:ssread:InvalidBinsparseComponent', ...
        'invalid BSP component name: %s', name) ;
end
if (strcmp (name, 'b') || strcmp (name, 'x'))
    if (isfield (bsp_problem, name))
        error ('SuiteSparse:ssread:DuplicateBinsparseComponent', ...
            'duplicate BSP component: %s', name) ;
    end
    bsp_problem.(name) = value ;
else
    if (~isfield (bsp_problem, 'aux'))
        bsp_problem.aux = struct ;
    elseif (isfield (bsp_problem.aux, name))
        error ('SuiteSparse:ssread:DuplicateBinsparseComponent', ...
            'duplicate BSP component: %s', name) ;
    end
    bsp_problem.aux.(name) = value ;
end
