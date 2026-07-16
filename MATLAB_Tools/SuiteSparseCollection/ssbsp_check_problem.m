function ssbsp_check_problem (Problem, bspfile)
%SSBSP_CHECK_PROBLEM compare a Binsparse file with a MATLAB Problem struct
%
%   ssbsp_check_problem (Problem, bspfile)
%
% Checks the primary matrix, explicit zero pattern, b, x, and supported aux
% components.  This is the component-level checker used by the collection BSP
% experiment; it does not reconstruct or compare the complete Problem metadata.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin ~= 2)
    error ('SuiteSparse:ssbsp_check_problem:InvalidArguments', ...
        'usage: ssbsp_check_problem (Problem, bspfile)') ;
end
check_problem (Problem, bspfile) ;


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
        actual = reshape (values, [bsp.ncols, bsp.nrows]).' ;
    case 'DMATC'
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

if (~isinteger (actual) || ~isinteger (expected))
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
