function sswrite (Problem, Master, arg3, arg4)
%SSWRITE write a Problem in Matrix Market, Rutherford/Boeing, or Binsparse format
% containing a set of files in Matrix Market, Rutherford/Boeing, or Binsparse
% format.  The MM and RB formats can be read from the files back into MATLAB via
% SSread.
% See http://sparse.tamu.edu for the SuiteSparse Matrix Collection home page.
% The Problem directory is optionally compressed via tar and gzip.  Arguments 3
% and 4, below, are optional and can appear in any order.
%
%    sswrite (Problem)          % Matrix Market format, no tar, use current dir.
%
% The following usages write the Problem into the Master/Group/Name directory,
% where Problem.name = 'Group/Name' is given in the Problem.
%
%    sswrite (Problem, Master)                 % Matrix Market, no tar
%    sswrite (Problem, Master, 'MM')           % ditto
%    sswrite (Problem, Master, 'RB')           % Rutherford/Boeing, no tar
%    sswrite (Problem, Master, 'BSP')          % Binsparse, no tar
%    sswrite (Problem, Master, 'tar')          % Matrix Market, with tar
%    sswrite (Problem, Master, 'MM', 'tar')    % ditto
%    sswrite (Problem, Master, 'RB', 'tar')    % Rutherford/Boeing, with tar
%    sswrite (Problem, Master, 'BSP', 'tar')   % Binsparse, with tar
%
% Problem is a struct, in the SuiteSparse Matrix format (see below).  Master is
% the top-level directory in which directory containing the problem will be
% placed.  Master defaults to the current working directory if not present (an
% empty string also means the current directory will be used).
%
% The following fields are always present in a SuiteSparse Matrix Problem:
%
%       name    the group directory and problem name (i.e. 'HB/arc130')
%
%       title   short descriptive title
%
%       A       an m-by-n sparse matrix, real or complex
%
%       id      an integer in the range of 1 to the # of problems in the
%               collection.  Once assigned, the id never changes.
%
%       date    a string containing the date the matrix was created, or added
%               to the collection if the creating date is unknown (but is likely
%               close to the creation date); empty string otherwise.
%
%       author  a string containing the name of the author; the computational
%               scientist who created the matrix from his or her application.
%               Empty string, or "author unknown" if unknown.
%
%       ed      a string containing the name of the editor/collector; the person
%               who acquired the matrix from the author, for inclusion in this
%               (or other) sparse matrix / graph collections.
%
%       kind    a string (i.e. 'directed graph') describing the type of problem;
%               a list of words from a well-defined set (see the SuiteSparse
%               Matrix Collection home page for a description).
%
% A Problem struct may also have the following optional fields.
%
%       Zeros   pattern of explicit zero entries in A as a binary m-by-n matrix.
%               These entries are provided by the matrix submittor with zero
%               numerical value.  MATLAB drops explicit zero entries from any
%               sparse matrix.  These entries can be important to the problem,
%               if (for example) the matrix is first in a series of matrices
%               arising in a nonlinear solver (those entries will become nonzero
%               later).  These entries are thus kept in the binary matrix Zeros.
%
%       b       right-hand-side, any size, real or complex, full or sparse
%
%       x       supposed solution to A*x=b, any size, real or complex, full or
%               sparse
%
%       notes   a character array with notes about the problem.
%
%       aux     a struct, containing auxiliary information.  The contents of
%               this struct are problem dependent, but its fields must not
%               end with '_[0-9]*' where [0-9]* means zero or more digits, nor
%               can there be an aux.b or aux.x matrix (there can be aux.b
%               and aux.x cells).  The aux struct must not include any structs,
%               just matrices and cell arrays of matrices.  The matrices in aux
%               can be full or sparse, and real, complex, or char.
%
% -------------------------------------------------
% for Problems written in Rutherford/Boeing format:
% -------------------------------------------------
%
% The Problem.name (including the full 'Group/Name'), date, author, and editor
% define the Rutherford/Boeing title (first line of the file), followed by a
% '|' in the 72nd column, and then up to 8 characters of the key.  If the key
% is an empty string, the Problem.id is used as the key.  The A and Zeros
% matrices are merged and written to this file.  The full name, title, id,
% kind, date, author, editor, and notes are written to a file of the same name
% as the primary file, but with a .txt extension.
%
% Additional Rutherford/Boeing files are created for b, x, and each sparse
% matrix in aux.  Full arrays are written using a tiny subset of the Matrix
% Market format.  The first line of the file is the header, either of:
%
%   %%MatrixMarket matrix array real general
%   %%MatrixMarket matrix array complex general
%
% The 2nd row contains the # of rows and columns, and subsequent lines contain
% one matrix entry each (two values if complex) listed in column-major order.
% No comments or blank lines are permitted.  The header is ignored when the
% matrix is read back in by ssread; the real/complex case is determined by how
% many entries appear on each line.  You can of course read these files with
% mread, the Matrix Market reader.  The header is added only to make it easier
% for functions *other* than ssread to read and understand the data (the file
% can be read in by mread, for example, but mread is not required).  Thus, a
% complete Rutherord/Boeing directory can be read/written via ssread/sswrite,
% without requiring the installation of mread/mwrite (in CHOLMOD), or
% mmread/mmread (M-file functions from NIST that read/write a Matrix Market
% file).
%
% ---------------------------------------------
% for Problems written in Matrix Market format:
% ---------------------------------------------
%
% The name, title, A, id, Zeros, kind, date, author, editor, and notes fields of
% the Problem are written to the primary Matrix Market file, with a .mtx
% extension.  Additional Matrix Market files are created for b (as name_b),
% x (as name_x), and each sparse or full matrix in aux.
%
% -------------------------------------
% for Problems written in Binsparse format:
% -------------------------------------
%
% The primary matrix, explicit zero pattern, b, x, and numeric aux matrices are
% written to a single Binsparse HDF5 file, with a .bsp.h5 extension.  Aux char
% arrays and cell arrays of strings are written as root-level HDF5 string
% datasets.  BSP output requires the Binsparse MATLAB bindings on the MATLAB
% path.
%
% -----------------
% for all formats:
% -----------------
%
% A matrix Problem.aux.whatever is written out as name_whatever.xxx, without
% the 'aux' part.  In BSP format, numeric matrices are groups and text
% components are root-level string datasets in the single name.bsp.h5 file.  In
% MM and RB formats, if Problem.aux.whatever is a char array, it is written as
% the file name_whatever.txt, with one line per row of the char array (trailing
% spaces in each line are not printed).  If aux.whatever is a cell array, each
% entry aux.whatever{i} is written as the file name_whatever_<i>.xxx
% (name_whatever_1.mtx, name_whatever_2.mtx, etc).  All files are placed in the
% single directory, given by the Problem.name (Group/Name, or 'HB/arc130' for
% example).  Each directory can only hold one MATLAB Problem struct of the
% SuiteSparse Matrix Collection.
%
% Example:
%
%   Problem = ssget ('HB/arc130')   % get the HB/arc130 MATLAB Problem
%   sswrite (Problem) ;             % write a MM version in current directory
%   sswrite (Problem, 'MM') ;       % write a MM version in MM/HB/arc130
%   sswrite (Problem, '', 'RB') ;   % write a RB version in current directory
%
% See also mwrite, mread, RBwrite, RBread, ssread, ssget, tar,
% write_binsparse_from_matlab

% Optionally uses the CHOLMOD mwrite mexFunction, for writing Problems in
% Matrix Market format.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

if (nargin < 2)
    % place the result in the current directory
    Master = '' ;
end
if (nargin < 3)
    arg3 = '' ;
end
if (nargin < 4)
    arg4 = '' ;
end
arg3 = lower (arg3) ;
arg4 = lower (arg4) ;

do_tar = (strcmp (arg3, 'tar') | strcmp (arg4, 'tar')) ;
RB = (strcmp (arg3, 'rb') | strcmp (arg4, 'rb')) ;
BSP = (strcmp (arg3, 'bsp') | strcmp (arg4, 'bsp')) ;
if (RB && BSP)
    error ('only one output format can be selected') ;
end

if (isfield (Problem, 'aux'))
    aux = Problem.aux ;
    auxfields = fields (aux) ;
else
    aux = struct ;
    auxfields = { } ;
end
validate_aux (auxfields, aux, BSP) ;

Master = regexprep (Master, '[\/\\]', '/') ;
if (~isempty (Master) && Master (end) ~= '/')
    Master = [Master '/'] ;
end

%-------------------------------------------------------------------------------
% create the problem directory.  Do not report any errors
%-------------------------------------------------------------------------------

t = find (Problem.name == '/') ;
group = Problem.name (1:t-1) ;
name = Problem.name (t+1:end) ;
groupdir = [Master group] ;
probdir = [groupdir '/' name] ;
probname = [probdir '/' name] ;

s = warning ('query', 'MATLAB:MKDIR:DirectoryExists') ;	    % get current state
warning ('off', 'MATLAB:MKDIR:DirectoryExists') ;
mkdir (groupdir) ;
mkdir (probdir) ;
warning (s) ;						    % restore state

%-------------------------------------------------------------------------------
% write the name, title, id, kind, date, author, ed, list of fields, and notes
%-------------------------------------------------------------------------------

cfile = [probname '.txt'] ;
if (RB)
    prefix = '%' ;
else
    prefix = '' ;
end
cf = fopen (cfile, 'w') ;
print_header (cf, prefix, Problem.name) ;
fprintf (cf, '%s [%s]\n', prefix, Problem.title) ;
fprintf (cf, '%s id: %d\n', prefix, Problem.id) ;
fprintf (cf, '%s date: %s\n', prefix, Problem.date) ;
fprintf (cf, '%s author: %s\n', prefix, Problem.author) ;
fprintf (cf, '%s ed: %s\n', prefix, Problem.ed) ;
fprintf (cf, '%s fields:', prefix) ;
s = fields (Problem) ;
for k = 1:length(s)
    fprintf (cf, ' %s', s {k}) ;
end
fprintf (cf, '\n') ;
if (isfield (Problem, 'aux'))
    fprintf (cf, '%s aux:', prefix) ;
    for k = 1:length(auxfields)
	fprintf (cf, ' %s', auxfields {k}) ;
    end
    fprintf (cf, '\n') ;
end
fprintf (cf, '%s kind: %s\n', prefix, Problem.kind) ;
print_separator (cf, prefix) ;
if (isfield (Problem, 'notes'))
    fprintf (cf, '%s notes:\n', prefix) ;
    for k = 1:size (Problem.notes,1)
	fprintf (cf, '%s %s\n', prefix, Problem.notes (k,:)) ;
    end
    print_separator (cf, prefix) ;
end
fclose(cf) ;

%-------------------------------------------------------------------------------
% write out the A and Z matrices to the primary file
%-------------------------------------------------------------------------------

A = Problem.A ;
[m n] = size (A) ;
if (~issparse (A) || m == 0 || n == 0)
    error ('A must be sparse and non-empty') ;
end
if (isfield (Problem, 'Zeros'))
    Z = Problem.Zeros ;
    if (any (size (A) ~= size (Z)) || ~isreal (Z) || ~issparse (Z))
	error ('Zeros must have same size as A, and be sparse and real') ;
    end
else
    Z = [ ] ;
end

% use the Problem.id number as the RB key
key = sprintf ('%d', Problem.id) ;

if (BSP)
    % write the supported Problem data in Binsparse form
    write_bsp_problem ([probname '.bsp.h5'], Problem) ;
    delete_if_exists (cfile) ;
    tar_problem (do_tar, probdir) ;
    return
elseif (RB)
    % write the files in Rutherford/Boeing form
    ptitle = [Problem.name '; ' Problem.date '; ' etal(Problem.author)] ;
    ptitle = [ptitle '; ed: ' etal(Problem.ed)] ;
    % note that b, g, and x are NOT written to the RB file
    RBwrite ([probname '.rb'], A, Z, ptitle, key) ;
else
    % write the files in Matrix Market form
    mwrite ([probname '.mtx'], A, Z, cfile) ;
    delete (cfile) ;    % the comments file has been included in the .mtx file.
end

%-------------------------------------------------------------------------------
% write out the b and x matrices as separate files
%-------------------------------------------------------------------------------

if (isfield (Problem, 'b'))
    write_component (probname, Problem.name, 'b', Problem.b, cfile, ...
	prefix, RB, [key 'b']) ;
end
if (isfield (Problem, 'x'))
    write_component (probname, Problem.name, 'x', Problem.x, cfile, ...
	prefix, RB, [key 'x']) ;
end

%-------------------------------------------------------------------------------
% write out each component of aux, each in a separate file
%-------------------------------------------------------------------------------

for k = 1:length(auxfields)
    what = auxfields {k} ;
    X = aux.(what) ;

    if (sscellstring (X) && Problem.id > 2776)

        % write out a cell array of strings as a single text file
        sstextwrite ([probname '_' what '.txt'], X) ;

    elseif (iscell (X))
	len = numel (X) ;
	for i = 1:len
	    % this problem includes a sequence of matrices in the new
	    % format (kind = 'sequence', and Problem.id > 1377).
	    write_component (probname, Problem.name, ...
		sprintf (fmt (i, len), what, i) , X{i}, cfile, ...
		prefix, RB, [key what(1) sprintf('%d', i)]) ;
	end
    else
	% This is a non-cell component of aux.  For an LP problem, this might
	% be c, lo, or hi.  For an Oberwolfach model reduction problem, this
	% might be M, C, or E.  For a graph in the Pajek collection, it could
	% be a vector 'year', with the publication of each article in the graph.
	% The possibilities are endless, and problem dependent.  Adding new
	% components to aux can be done without modifying ssread or sswrite.
	write_component (probname, Problem.name, what, X, cfile, ...
	    prefix, RB, [key what]) ;
    end
end

%-------------------------------------------------------------------------------
% tar up the result, if requested
%-------------------------------------------------------------------------------

tar_problem (do_tar, probdir) ;



%-------------------------------------------------------------------------------
% fmt
%-------------------------------------------------------------------------------

function s = fmt (i, len)
% fmt: determine the format to use for the name of component in a aux.cell array
if (len < 10)
    s = '%s_%d' ;		% x1 through x9
elseif (len < 100)
    if (i < 10)
	s = '%s_0%d' ;		% x01 through x09
    else
	s = '%s_%d' ;		% x10 through x99
    end
else
    if (i < 10)
	s = '%s_00%d' ;		% x001 through x009
    elseif (i < 100)
	s = '%s_0%d' ;		% x010 through x099
    else
	s = '%s_%d' ;		% x100 through x999
    end
end


%-------------------------------------------------------------------------------
% write_component
%-------------------------------------------------------------------------------

function write_component (probname, name, what, X, cfile, prefix, RB, key)
% write_component: write out a single component of the Problem to a file
if (isempty (X))
    % empty matrices (one or more zero dimensions) are not supported
    error (['invalid component: ' what ' (cannot be empty)']) ;
elseif (ischar (X))
    % Write out a char array as a text file.  Remove trailing spaces from the
    % strings.  Keep the leading spaces; they might be significant.
    ff = fopen ([probname '_' what '.txt'], 'w') ;
    for i = 1:size (X,1)
	fprintf (ff, '%s\n', deblank (X (i,:))) ;
    end
    fclose (ff) ;
elseif (RB)
    % write out a full or sparse matrix in Rutherford/Boeing format
    if (issparse (X))
	% write out the component as a Rutherford/Boeing sparse matrix
	RBwrite ([probname '_' what '.rb'], X, [ ], [name '_' what], key) ;
    else
	% Write out a full matrix in column oriented form,
	% using a tiny subset of the Matrix Market format for full matrices.
	ssfull_write ([probname '_' what '.mtx'], X) ;
    end
else
    % write out the component in Matrix Market format (full or sparse)
    cf = fopen (cfile, 'w') ;
    print_header (cf, prefix, name, what) ;
    print_separator (cf, prefix) ;
    fclose(cf) ;
    mwrite ([probname '_' what '.mtx'], X, cfile) ;
    delete (cfile) ;
end


%-------------------------------------------------------------------------------
% tar_problem
%-------------------------------------------------------------------------------

function tar_problem (do_tar, probdir)
% tar_problem: tar up the result, if requested
if (do_tar)
    try
	tar ([probdir '.tar.gz'], probdir) ;
	rmdir (probdir, 's') ;
    catch
	warning ('SuiteSparse:sswrite', ...
	    'unable to create tar file; directly left uncompressed') ;
    end
end


%-------------------------------------------------------------------------------
% write_bsp_problem
%-------------------------------------------------------------------------------

function write_bsp_problem (filename, Problem)
% write_bsp_problem: write the supported Problem data to one Binsparse HDF5 file
compression = 9 ;
Problem = expand_bsp_aux_cells (Problem) ;
if (exist ('write_binsparse_from_matlab', 'file') == 3)
    write_binsparse_from_matlab (Problem, filename, 'COO', [ ], compression) ;
elseif (exist ('generate_bsp_from_ssmc', 'file') == 2)
    generate_bsp_from_ssmc (Problem, filename, 'COO', compression) ;
else
    error ('BSP output requires the Binsparse MATLAB bindings on the path') ;
end


%-------------------------------------------------------------------------------
% expand_bsp_aux_cells
%-------------------------------------------------------------------------------

function Problem = expand_bsp_aux_cells (Problem)
% expand_bsp_aux_cells: map aux.foo{i} to aux.foo_i using sswrite's names
if (~isfield (Problem, 'aux'))
    return
end
aux = Problem.aux ;
aux2 = struct ;
auxfields = fields (aux) ;
for k = 1:length (auxfields)
    what = auxfields {k} ;
    X = aux.(what) ;
    if (iscell (X) && ~iscellstr (X))
        len = numel (X) ;
        for i = 1:len
            aux2.(sprintf (fmt (i, len), what, i)) = X {i} ;
        end
    else
        aux2.(what) = X ;
    end
end
Problem.aux = aux2 ;


%-------------------------------------------------------------------------------
% validate_aux
%-------------------------------------------------------------------------------

function validate_aux (auxfields, aux, BSP)
% validate_aux: ensure aux field names map unambiguously to filenames
for k = 1:length (auxfields)
    what = auxfields {k} ;
    X = aux.(what) ;
    if (BSP && (~iscell (X) || iscellstr (X)) && ...
            any (strcmp (what, {'b', 'x', 'values', 'indices_0', ...
            'indices_1', 'pointers_to_1'})))
        % These names are written directly at the HDF5 root, where they would
        % collide with a primary matrix dataset or a Problem.b/Problem.x group.
        error (['invalid BSP aux component: ' what]) ;
    end
    if (~iscell (X) && (strcmp (what, 'b') || strcmp (what, 'x')))
        % aux.b or aux.x would get written out with the same filename as the
        % Problem.b and Problem.x matrices, and read back in by ssread as
        % Problem.b and Problem.x instead of aux.b and aux.x.
        error (['invalid aux component: ' what]) ;
    end
    if (~isempty (regexp (what, '_[0-9]*\>', 'once')))
        % aux.whatever_42 would be written as the file name_whatever_42, which
        % would be intrepretted as aux.whatever{42} when read back in by ssread.
        error (['invalid aux component: ' what]) ;
    end
end


%-------------------------------------------------------------------------------
% delete_if_exists
%-------------------------------------------------------------------------------

function delete_if_exists (filename)
% delete_if_exists: delete a file if it exists
if (exist (filename, 'file'))
    delete (filename) ;
end


%-------------------------------------------------------------------------------
% print_separator
%-------------------------------------------------------------------------------

function print_separator (cf, prefix)
% print_separator: print a separator line in the comment file
fprintf (cf, '%s---------------------------------------', prefix) ;
fprintf (cf, '----------------------------------------\n') ;


%-------------------------------------------------------------------------------
% print_header
%-------------------------------------------------------------------------------

function print_header (cf, prefix, name, what)
% print_header: print the header to the comment file
print_separator (cf, prefix) ;
fprintf (cf, '%s SuiteSparse Matrix Collection, Tim Davis\n', prefix) ;
fprintf (cf, '%s https://sparse.tamu.edu/%s\n', prefix, name) ;
fprintf (cf, '%s name: %s', prefix, name) ;
if (nargin > 3)
    fprintf (cf, ' : %s matrix', what) ;
end
fprintf (cf, '\n') ;


%-------------------------------------------------------------------------------
% etal
%-------------------------------------------------------------------------------

function s = etal(name)
% etal: change a long list of authors to first author 'et al.'
t = find (name == ',') ;
if (length (t) > 1)
    s = [name(1:t(1)-1) ' et al.'] ;
else
    s = name ;
end
