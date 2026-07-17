function ssexport (list, check, tmp, formats, topdir)
%SSEXPORT export to Matrix Market, Rutherford/Boeing, or Binsparse formats
%
% Example:
%   ssexport ;                    % export the entire collection to MM, RB, BSP
%   ssexport (list) ;             % export selected matrices to MM, RB, BSP
%   ssexport (list, 'check') ;    % also read them back in, to check
%   ssexport (list, 'check', '', {'BSP'}) ;             % just export to BSP
%   ssexport (list, 'check', '', {'MM', 'RB', 'BSP'}) ; % export to all & check
%
% If the list is empty, all matrices in the collection are exported.
% A 3rd argument tmp changes the tmp directory for ssread.
% The optional 4th argument selects the output formats.  It may be a character
% vector or a cell array containing 'MM', 'RB', and/or 'BSP'.  The default is
% {'MM', 'RB'} for backwards compatibility.  The optional 5th argument changes
% the output directory from sslocation, primarily for testing.
%
% See also ssget, sswrite, ssread, ssbsp_check_problem, RBio, mwrite.

% SuiteSparseCollection, Copyright (c) 2006-2026, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

%-------------------------------------------------------------------------------
% get the input arguments
%-------------------------------------------------------------------------------

index = ssget ;
nmat = length (index.nrows) ;

if (nargin < 1 || isempty (list))
    list = 1:nmat ;
end

check = ((nargin > 1) && strcmp (check, 'check')) ;

if (nargin < 3)
    tmp = '' ;
end

if (nargin < 4 || isempty (formats))
    formats = {'MM', 'RB', 'BSP'} ;
elseif (ischar (formats))
    formats = {formats} ;
elseif (~iscell (formats) || ~all (cellfun (@ischar, formats)))
    error ('SuiteSparse:ssexport:InvalidFormats', ...
        'formats must be a character vector or cell array of character vectors') ;
end

for k = 1:numel (formats)
    formats{k} = upper (formats{k}) ;
    if (~any (strcmp (formats{k}, {'MM', 'RB', 'BSP'})))
        error ('SuiteSparse:ssexport:InvalidFormat', ...
            'unsupported export format: %s', formats{k}) ;
    end
end
if (numel (unique (formats)) ~= numel (formats))
    error ('SuiteSparse:ssexport:DuplicateFormat', ...
        'export formats must not contain duplicates') ;
end

if (any (strcmp (formats, 'BSP')))
    have_mex_writer = (exist ('write_binsparse_from_matlab', 'file') == 3) ;
    have_matlab_writer = (exist ('generate_bsp_from_ssmc', 'file') == 2) ;
    if (~have_mex_writer && ~have_matlab_writer)
        error ('SuiteSparse:ssexport:MissingBinsparseWriter', ...
            'BSP export requires the Binsparse MATLAB bindings on the path') ;
    end
    if (check)
        reader = which ('binsparse_read') ;
        [~, ~, reader_extension] = fileparts (reader) ;
        if (isempty (reader) || ...
                ~strcmpi (reader_extension, ['.' mexext]))
            error ('SuiteSparse:ssexport:MissingBinsparseReader', ...
                'checking BSP output requires the binsparse_read MEX function') ;
        end
    end
end

%-------------------------------------------------------------------------------
% determine the top-level directory to use
%-------------------------------------------------------------------------------

if (nargin < 5 || isempty (topdir))
    topdir = sslocation ;
end
fprintf ('\nExport to topdir: %s\ncheck: %d\ntmp: %s\n', topdir, check, tmp) ;
fprintf ('formats:') ;
for k = 1:numel (formats)
    fprintf (' %s', formats{k}) ;
end
fprintf ('\n') ;
for id = list
    fprintf ('%4d : %s/%s\n', id, index.Group {id}, index.Name {id}) ;
end

%-------------------------------------------------------------------------------
% export the matrices
%-------------------------------------------------------------------------------

for id = list

    % get the MATLAB version
    clear Problem
    Problem = ssget (id, index) ;
    disp (Problem) ;
    if (isfield (Problem, 'aux'))
        disp (Problem.aux) ;
    end

    % create each requested version
    for k = 1:numel (formats)
        format = formats{k} ;
        fprintf ('Exporting to %s format ...\n', format) ;
        outputdir = fullfile (topdir, format) ;
        if (nnz (Problem.A) < 1e8)
            sswrite (Problem, outputdir, format, 'tar') ;
        else
            % the MATLAB tar has problems with huge files
            fprintf ('File too big for MATLAB tar\n') ;
            sswrite (Problem, outputdir, format) ;
        end
    end

    % check each newly written version
    if (check)
        for k = 1:numel (formats)
            format = formats{k} ;
            fprintf ('Reading %s format ...\n', format) ;
            if (strcmp (format, 'BSP'))
                [bspfile, bsp_cleanup] = bsp_check_file (...
                    topdir, Problem, tmp) ;                        %#ok<ASGLU>
                ssbsp_check_problem (Problem, bspfile) ;
                clear bsp_cleanup
                fprintf ('Comparing MATLAB and BSP format ... OK.\n') ;
                continue
            end
            try
                problem_path = fullfile (topdir, format, Problem.name) ;
                if (isempty (tmp))
                    P2 = ssread (problem_path) ;
                else
                    P2 = ssread (problem_path, tmp) ;
                end
            catch me
                % The Problem may be too large for two copies to be in the
                % MATLAB workspace at the same time.  This is not an error,
                % but it means that the Problem cannot be checked.
                P2 = [ ] ;
                fprintf ('Unable to read %s/%s\n', format, Problem.name) ;
                fprintf ('%s\n', me.message) ;
            end
            fprintf ('Comparing MATLAB and %s format ...\n', format) ;
            if (~isempty (P2) && ~isequal (Problem, P2))
                disp (Problem) ;
                disp (P2) ;
                warning ('%s version mismatch: %s\n', format, Problem.name) ;
                e = norm (Problem.A - P2.A, 1) ;
                a = norm (Problem.A, 1) ;
                fprintf ('norm (A1-A2,1): %g  relative: %g\n', e, e/a) ;
            end
            clear P2
        end
        fprintf ('OK.\n') ;
    end
end


%-------------------------------------------------------------------------------
% bsp_check_file
%-------------------------------------------------------------------------------

function [bspfile, cleanup] = bsp_check_file (topdir, Problem, tmp)
% Locate uncompressed BSP output, or extract its archive for checking.

t = find (Problem.name == '/') ;
name = Problem.name (t(end)+1:end) ;
probdir = fullfile (topdir, 'BSP', Problem.name) ;
bspfile = fullfile (probdir, [name '.bsp.h5']) ;
cleanup = [ ] ;
if (exist (bspfile, 'file') == 2)
    return
end

archive = [probdir '.tar.gz'] ;
if (exist (archive, 'file') ~= 2)
    error ('SuiteSparse:ssexport:MissingBinsparseOutput', ...
        'unable to find BSP output for %s', Problem.name) ;
end

if (isempty (tmp))
    extractdir = tempname ;
else
    if (~exist (tmp, 'dir'))
        mkdir (tmp) ;
    end
    extractdir = tempname (tmp) ;
end
mkdir (extractdir) ;
cleanup = onCleanup (@() remove_extract_dir (extractdir)) ;
untar (archive, extractdir) ;

files = dir (fullfile (extractdir, '**', '*.bsp.h5')) ;
if (isempty (files))
    error ('SuiteSparse:ssexport:InvalidBinsparseArchive', ...
        'BSP archive for %s contains no .bsp.h5 file', Problem.name) ;
end
expected = strcmp ({files.name}, [name '.bsp.h5']) ;
if (nnz (expected) ~= 1)
    error ('SuiteSparse:ssexport:InvalidBinsparseArchive', ...
        'BSP archive for %s does not contain one expected file', Problem.name) ;
end
file = files (expected) ;
bspfile = fullfile (file.folder, file.name) ;


%-------------------------------------------------------------------------------
% remove_extract_dir
%-------------------------------------------------------------------------------

function remove_extract_dir (extractdir)
% Remove temporary files created while checking a BSP archive.
if (exist (extractdir, 'dir'))
    rmdir (extractdir, 's') ;
end
