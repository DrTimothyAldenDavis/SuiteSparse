function results = ssbsp_test_collection (list, varargin)
%SSBSP_TEST_COLLECTION test BSP export for SuiteSparse Matrix Collection
%
%   results = ssbsp_test_collection ;
%   results = ssbsp_test_collection (list) ;
%   results = ssbsp_test_collection (..., 'option', value, ...) ;
%
% This experiment loads each requested SuiteSparse Matrix Collection Problem
% with ssget, writes it with sswrite(...,'BSP'), reads the resulting Binsparse
% file back into MATLAB, and checks that the numeric and text contents match
% the original Problem exactly.  BSP files are deleted after each matrix by
% default.
%
% Options:
%   'WorkDir'              directory for temporary BSP output
%   'LogFile'              tab-separated progress log
%   'ResultFile'           MAT-file checkpoint with the results struct
%   'DeleteBSP'            delete each generated BSP file (default true)
%   'DeleteDownloadedMat'  delete ssget .mat files downloaded by this run
%   'Resume'               skip ids already marked ok in ResultFile
%   'StopOnFailure'        stop on first failed matrix
%   'MaxMatrices'          only run the first N ids in list
%   'MaxNnz'               skip matrices with more than this many entries
%   'MinFreeGB'            fail before a matrix if WorkDir has less free space
%
% BSP output requires the Binsparse MATLAB bindings on the MATLAB path.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin < 1)
    list = [ ] ;
end

opts = parse_options (varargin{:}) ;
require_experiment_functions ;

if (~exist (opts.WorkDir, 'dir'))
    mkdir (opts.WorkDir) ;
end
if (isempty (opts.LogFile))
    opts.LogFile = fullfile (opts.WorkDir, 'ssbsp_test_collection.log') ;
end
if (isempty (opts.ResultFile))
    opts.ResultFile = fullfile (opts.WorkDir, ...
        'ssbsp_test_collection_results.mat') ;
end

index = ssget ;
nmat = length (index.nrows) ;
if (isempty (list))
    list = 1:nmat ;
end
list = list (:)' ;
if (~isempty (opts.MaxMatrices))
    list = list (1:min (length (list), opts.MaxMatrices)) ;
end

results = empty_results (0) ;
if (opts.Resume && exist (opts.ResultFile, 'file'))
    checkpoint = load (opts.ResultFile) ;
    if (isfield (checkpoint, 'results'))
        results = checkpoint.results ;
    end
end

fid = fopen (opts.LogFile, 'a') ;
if (fid < 0)
    error ('unable to open log file: %s', opts.LogFile) ;
end
log_cleanup = onCleanup (@() fclose (fid)) ;

fprintf (fid, '# %s ssbsp_test_collection start, %d requested ids\n', ...
    datestr (now, 31), length (list)) ;
fprintf ('BSP collection experiment work dir: %s\n', opts.WorkDir) ;
fprintf ('Log file: %s\n', opts.LogFile) ;
fprintf ('Result file: %s\n', opts.ResultFile) ;

for kk = 1:length (list)

    id = list (kk) ;
    requested_name = sprintf ('%s/%s', index.Group{id}, index.Name{id}) ;
    existing = find_result (results, id) ;
    if (opts.Resume && existing > 0 && strcmp (results (existing).status, 'ok'))
        fprintf ('%5d/%5d id %d %s: skip existing ok\n', ...
            kk, length (list), id, requested_name) ;
        continue ;
    end

    rec = empty_results (1) ;
    rec.id = id ;
    rec.name = requested_name ;
    rec.matfile = collection_matfile (index, id) ;
    rec.matfile_preexisting = (exist (rec.matfile, 'file') == 2) ;

    t = tic ;
    fprintf ('%5d/%5d id %d %s\n', kk, length (list), id, requested_name) ;

    try
        if (index.nnz (id) > opts.MaxNnz)
            rec.status = 'skip' ;
            rec.identifier = 'ssbsp_test_collection:MaxNnz' ;
            rec.message = sprintf ('nnz %.0f exceeds MaxNnz %.0f', ...
                index.nnz (id), opts.MaxNnz) ;
        else
            check_free_space (opts) ;

            Problem = ssget (id, index) ;                                  %#ok
            rec.name = Problem.name ;
            rec.bspfile = bsp_filename (opts.WorkDir, Problem.name) ;

            sswrite (Problem, opts.WorkDir, 'BSP') ;
            info = dir (rec.bspfile) ;
            if (~isempty (info))
                rec.bsp_bytes = info.bytes ;
            end

            check_problem (Problem, rec.bspfile) ;
            rec.status = 'ok' ;
        end

    catch me
        rec.status = 'fail' ;
        rec.identifier = me.identifier ;
        rec.message = me.message ;
    end

    rec.seconds = toc (t) ;
    results = store_result (results, rec) ;
    log_result (fid, rec) ;
    save (opts.ResultFile, 'results', 'list', 'opts') ;

    cleanup_generated_files (opts, rec) ;
    clear Problem

    if (strcmp (rec.status, 'fail') && opts.StopOnFailure)
        error ('ssbsp_test_collection:Failed', 'id %d %s failed: %s', ...
            rec.id, rec.name, rec.message) ;
    end
end

fprintf (fid, '# %s ssbsp_test_collection done\n', datestr (now, 31)) ;


%-------------------------------------------------------------------------------
% options and bookkeeping
%-------------------------------------------------------------------------------

function opts = parse_options (varargin)

opts.WorkDir = fullfile (tempdir, 'ssbsp_test_collection') ;
opts.LogFile = '' ;
opts.ResultFile = '' ;
opts.DeleteBSP = true ;
opts.DeleteDownloadedMat = false ;
opts.Resume = true ;
opts.StopOnFailure = false ;
opts.MaxMatrices = [ ] ;
opts.MaxNnz = Inf ;
opts.MinFreeGB = 0 ;

if (mod (length (varargin), 2) ~= 0)
    error ('options must be name/value pairs') ;
end

for k = 1:2:length (varargin)
    name = varargin{k} ;
    value = varargin{k+1} ;
    if (~ischar (name))
        error ('option names must be character vectors') ;
    end
    switch (lower (name))
        case 'workdir'
            opts.WorkDir = value ;
        case 'logfile'
            opts.LogFile = value ;
        case 'resultfile'
            opts.ResultFile = value ;
        case 'deletebsp'
            opts.DeleteBSP = logical (value) ;
        case 'deletedownloadedmat'
            opts.DeleteDownloadedMat = logical (value) ;
        case 'resume'
            opts.Resume = logical (value) ;
        case 'stoponfailure'
            opts.StopOnFailure = logical (value) ;
        case 'maxmatrices'
            opts.MaxMatrices = value ;
        case 'maxnnz'
            opts.MaxNnz = value ;
        case 'minfreegb'
            opts.MinFreeGB = value ;
        otherwise
            error ('unknown option: %s', name) ;
    end
end


function require_experiment_functions

required = {'ssget', 'sswrite', 'binsparse_read', ...
            'write_binsparse_from_matlab'} ;
for k = 1:length (required)
    if (exist (required{k}, 'file') == 0)
        error ('%s is required on the MATLAB path', required{k}) ;
    end
end


function results = empty_results (n)

record = struct ( ...
    'id', [ ], ...
    'name', '', ...
    'status', '', ...
    'seconds', [ ], ...
    'identifier', '', ...
    'message', '', ...
    'matfile', '', ...
    'matfile_preexisting', false, ...
    'bspfile', '', ...
    'bsp_bytes', 0) ;
results = repmat (record, n, 1) ;


function k = find_result (results, id)

k = 0 ;
for j = 1:length (results)
    if (isequal (results (j).id, id))
        k = j ;
        return
    end
end


function results = store_result (results, rec)

k = find_result (results, rec.id) ;
if (k == 0)
    results (end+1, 1) = rec ;
else
    results (k) = rec ;
end


function log_result (fid, rec)

fprintf (fid, '%s\t%d\t%s\t%s\t%.6g\t%d\t%s\t%s\n', ...
    datestr (now, 31), rec.id, rec.name, rec.status, rec.seconds, ...
    rec.bsp_bytes, clean_log_text (rec.identifier), ...
    clean_log_text (rec.message)) ;
fprintf ('    %s in %.3g sec', rec.status, rec.seconds) ;
if (~isempty (rec.message))
    fprintf (': %s', rec.message) ;
end
fprintf ('\n') ;


function s = clean_log_text (s)

s = strrep (s, sprintf ('\n'), ' ') ;
s = strrep (s, sprintf ('\t'), ' ') ;


function file = collection_matfile (index, id)

params = ssget_defaults ;
file = fullfile (params.topdir, 'mat', index.Group{id}, ...
    [index.Name{id} '.mat']) ;


function file = bsp_filename (workdir, name)

t = find (name == '/') ;
group = name (1:t-1) ;
matrix = name (t+1:end) ;
file = fullfile (workdir, group, matrix, [matrix '.bsp.h5']) ;


function cleanup_generated_files (opts, rec)

if (opts.DeleteBSP && ~isempty (rec.bspfile) && exist (rec.bspfile, 'file'))
    delete (rec.bspfile) ;
    remove_empty_dir (fileparts (rec.bspfile)) ;
    remove_empty_dir (fileparts (fileparts (rec.bspfile))) ;
end

if (opts.DeleteDownloadedMat && ~rec.matfile_preexisting && ...
        ~isempty (rec.matfile) && exist (rec.matfile, 'file'))
    delete (rec.matfile) ;
    remove_empty_dir (fileparts (rec.matfile)) ;
end


function remove_empty_dir (directory)

if (~exist (directory, 'dir'))
    return
end
files = dir (directory) ;
if (length (files) == 2)
    try
        rmdir (directory) ;
    catch
    end
end


function check_free_space (opts)

if (opts.MinFreeGB <= 0)
    return
end
[status, output] = system (sprintf ('df -Pk "%s" | tail -1', opts.WorkDir)) ;
if (status ~= 0)
    return
end
parts = regexp (strtrim (output), '\s+', 'split') ;
if (length (parts) >= 4)
    free_gb = str2double (parts{4}) / 1024 / 1024 ;
    if (free_gb < opts.MinFreeGB)
        error ('ssbsp_test_collection:LowDiskSpace', ...
            'only %.3g GB free in %s', free_gb, opts.WorkDir) ;
    end
end


%-------------------------------------------------------------------------------
% problem comparison
%-------------------------------------------------------------------------------

function check_problem (Problem, bspfile)

if (exist (bspfile, 'file') ~= 2)
    error ('BSP file was not created: %s', bspfile) ;
end

Zeros = [ ] ;
if (isfield (Problem, 'Zeros'))
    Zeros = Problem.Zeros ;
end
check_bsp_numeric (bspfile, '', Problem.A, 'A', Zeros) ;

if (isfield (Problem, 'b') && ~isempty (Problem.b))
    check_component (bspfile, 'b', Problem.b, 'b') ;
end
if (isfield (Problem, 'x') && ~isempty (Problem.x))
    check_component (bspfile, 'x', Problem.x, 'x') ;
end

if (isfield (Problem, 'aux') && isstruct (Problem.aux))
    names = fieldnames (Problem.aux) ;
    for k = 1:length (names)
        name = names{k} ;
        check_component (bspfile, name, Problem.aux.(name), ...
            ['aux.' name]) ;
    end
end


function check_component (bspfile, name, value, label)

if (isempty (value))
    error ('component %s is empty and was not represented', label) ;
end

if (is_text_value (value))
    check_text_dataset (bspfile, name, value, label) ;
elseif (iscell (value))
    len = numel (value) ;
    for k = 1:len
        child = component_name (name, k, len) ;
        check_component (bspfile, child, value{k}, ...
            sprintf ('%s{%d}', label, k)) ;
    end
elseif (islogical (value))
    check_bsp_numeric (bspfile, name, double (value), label, [ ]) ;
elseif (issparse (value) || isnumeric (value))
    check_bsp_numeric (bspfile, name, value, label, [ ]) ;
else
    error ('component %s has unsupported type %s', label, class (value)) ;
end


function ok = is_text_value (value)

ok = ischar (value) || iscellstr (value) || ...
    (exist ('isstring', 'builtin') && isstring (value)) ;


function check_text_dataset (bspfile, name, expected, label)

dataset = ['/' name] ;
try
    actual = h5read (bspfile, dataset) ;
catch me
    error ('missing text dataset %s for %s: %s', dataset, label, me.message) ;
end
actual = h5_text_to_cellstr (actual) ;
expected = expected_text (expected) ;
if (~isequal (actual, expected))
    error ('text mismatch for %s', label) ;
end


function value = expected_text (value)

if (ischar (value))
    if (isempty (value) && size (value, 1) == 0)
        value = {''} ;
    else
        chars = value ;
        value = cell (size (chars, 1), 1) ;
        for k = 1:size (chars, 1)
            value{k} = chars (k, :) ;
        end
    end
elseif (iscellstr (value))
    value = value (:) ;
elseif (exist ('isstring', 'builtin') && isstring (value))
    value = cellstr (value (:)) ;
else
    error ('unexpected text type') ;
end
value = normalize_text_cells (value) ;


function value = h5_text_to_cellstr (value)

if (iscell (value))
    value = value (:) ;
elseif (ischar (value))
    value = expected_text (value) ;
elseif (exist ('isstring', 'builtin') && isstring (value))
    value = cellstr (value (:)) ;
else
    error ('unexpected HDF5 text type') ;
end
value = normalize_text_cells (value) ;


function value = normalize_text_cells (value)

for k = 1:numel (value)
    value{k} = reshape (char (value{k}), 1, [ ]) ;
end


function check_bsp_numeric (bspfile, group, expected, label, Zeros)

try
    if (isempty (group))
        bsp = binsparse_read (bspfile) ;
    else
        bsp = binsparse_read (bspfile, group) ;
    end
catch me
    error ('unable to read BSP matrix %s: %s', label, me.message) ;
end

if (issparse (expected))
    check_sparse_bsp (bsp, expected, label, Zeros) ;
else
    actual = dense_bsp_to_matlab (bsp, label) ;
    if (~numeric_equal (actual, expected))
        error ('numeric mismatch for %s', label) ;
    end
end


function check_sparse_bsp (bsp, expected, label, Zeros)

if (bsp.nrows ~= size (expected, 1) || bsp.ncols ~= size (expected, 2))
    error ('size mismatch for %s', label) ;
end

[rows, cols, values] = sparse_bsp_entries (bsp, label) ;
if (isempty (Zeros))
    zero_mask = false (size (values)) ;
else
    zero_mask = (values == 0) ;
end

if (~isempty (Zeros))
    compare_pattern (rows (zero_mask), cols (zero_mask), Zeros, ...
        [label '.Zeros']) ;
end
compare_sparse_entries (rows (~zero_mask), cols (~zero_mask), ...
    values (~zero_mask), expected, label) ;


function [rows, cols, values] = sparse_bsp_entries (bsp, label)

fmt = upper (bsp.format) ;
values = bsp.values (:) ;
switch (fmt)
    case {'COO', 'COOR'}
        rows = double (bsp.indices_0 (:)) + 1 ;
        cols = double (bsp.indices_1 (:)) + 1 ;
    case 'CSC'
        colptr = double (bsp.pointers_to_1 (:)) ;
        rows = double (bsp.indices_1 (:)) + 1 ;
        cols = zeros (length (values), 1) ;
        for j = 1:bsp.ncols
            p = (colptr (j) + 1):colptr (j + 1) ;
            cols (p) = j ;
        end
    case 'CSR'
        rowptr = double (bsp.pointers_to_1 (:)) ;
        cols = double (bsp.indices_1 (:)) + 1 ;
        rows = zeros (length (values), 1) ;
        for i = 1:bsp.nrows
            p = (rowptr (i) + 1):rowptr (i + 1) ;
            rows (p) = i ;
        end
    otherwise
        error ('unexpected sparse BSP format for %s: %s', label, bsp.format) ;
end

if (isfield (bsp, 'is_iso') && bsp.is_iso)
    values = repmat (values (1), length (rows), 1) ;
end


function compare_sparse_entries (rows, cols, values, expected, label)

[erows, ecols, evalues] = find (expected) ;
erows = double (erows (:)) ;
ecols = double (ecols (:)) ;
evalues = evalues (:) ;

if (length (values) ~= length (evalues))
    error ('stored entry count mismatch for %s', label) ;
end

[rows, cols, values] = sort_entries (rows, cols, values) ;
[erows, ecols, evalues] = sort_entries (erows, ecols, evalues) ;

if (~isequal (rows, erows) || ~isequal (cols, ecols) || ...
        ~numeric_equal (values, evalues))
    error ('sparse entry mismatch for %s', label) ;
end


function compare_pattern (rows, cols, expected, label)

[erows, ecols] = find (expected) ;
rows = double (rows (:)) ;
cols = double (cols (:)) ;
erows = double (erows (:)) ;
ecols = double (ecols (:)) ;

if (length (rows) ~= length (erows))
    error ('pattern count mismatch for %s', label) ;
end

[rows, cols] = sort_pattern (rows, cols) ;
[erows, ecols] = sort_pattern (erows, ecols) ;
if (~isequal (rows, erows) || ~isequal (cols, ecols))
    error ('pattern mismatch for %s', label) ;
end


function [rows, cols, values] = sort_entries (rows, cols, values)

[~, p] = sortrows ([double(rows(:)) double(cols(:))]) ;
rows = double (rows (p)) ;
cols = double (cols (p)) ;
values = values (p) ;


function [rows, cols] = sort_pattern (rows, cols)

[~, p] = sortrows ([double(rows(:)) double(cols(:))]) ;
rows = double (rows (p)) ;
cols = double (cols (p)) ;


function actual = dense_bsp_to_matlab (bsp, label)

fmt = upper (bsp.format) ;
values = bsp.values (:) ;
if (isfield (bsp, 'is_iso') && bsp.is_iso)
    values = repmat (values (1), double (bsp.nrows) * double (bsp.ncols), 1) ;
end
switch (fmt)
    case 'DMAT'
        actual = reshape (values, [bsp.nrows, bsp.ncols]) ;
    case 'DVEC'
        actual = reshape (values, [bsp.nrows, 1]) ;
    otherwise
        error ('unexpected dense BSP format for %s: %s', label, bsp.format) ;
end


function ok = numeric_equal (actual, expected)

if (~isequal (size (actual), size (expected)))
    ok = false ;
    return
end

if (isfloat (actual) || isfloat (expected) || ...
        ~isinteger (actual) || ~isinteger (expected))
    ok = isequaln (double (actual), double (expected)) ;
else
    ok = isequaln (actual, cast (expected, class (actual))) && ...
        isequaln (cast (actual, class (expected)), expected) ;
end


function name = component_name (prefix, index, count)

if (count < 10)
    pattern = '%s_%d' ;
elseif (count < 100)
    if (index < 10)
        pattern = '%s_0%d' ;
    else
        pattern = '%s_%d' ;
    end
else
    if (index < 10)
        pattern = '%s_00%d' ;
    elseif (index < 100)
        pattern = '%s_0%d' ;
    else
        pattern = '%s_%d' ;
    end
end
name = sprintf (pattern, prefix, index) ;
