function ssbsp_test_ssread
%SSBSP_TEST_SSREAD test Binsparse sswrite/ssread integration

% SuiteSparseCollection, Copyright (c) 2006-2026, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (exist ('write_binsparse_from_matlab', 'file') ~= 3 && ...
        exist ('generate_bsp_from_ssmc', 'file') ~= 2)
    error ('ssbsp_test_ssread:MissingWriter', ...
        'Binsparse MATLAB writer is not on the path') ;
end
reader = which ('binsparse_read') ;
[~, ~, reader_extension] = fileparts (reader) ;
converter = which ('convert_to_problem_struct') ;
if (isempty (reader) || ~strcmpi (reader_extension, ['.' mexext]) || ...
        isempty (converter))
    error ('ssbsp_test_ssread:MissingReader', ...
        'Binsparse MATLAB reader and converter are not on the path') ;
end

Problem = synthetic_problem ;
work = tempname ;
mkdir (work) ;
cleanup = onCleanup (@() remove_workdir (work)) ;

plain = fullfile (work, 'plain') ;
sswrite (Problem, plain, 'BSP') ;
actual = ssread (fullfile (plain, Problem.name)) ;
assert_problem_equal (actual, Problem, 'uncompressed BSP roundtrip') ;

archived = fullfile (work, 'archived') ;
sswrite (Problem, archived, 'BSP', 'tar') ;
problem_path = fullfile (archived, Problem.name) ;
assert (exist ([problem_path '.tar.gz'], 'file') == 2, ...
    'BSP archive was not created') ;
actual = ssread (problem_path, fullfile (work, 'extract')) ;
assert_problem_equal (actual, Problem, 'archived BSP roundtrip') ;

fprintf ('ssbsp_test_ssread: all tests passed\n') ;


%-------------------------------------------------------------------------------
% synthetic_problem
%-------------------------------------------------------------------------------

function Problem = synthetic_problem

Problem = struct ;
Problem.name = 'Test/ssread' ;
Problem.title = 'Binsparse ssread integration test' ;
Problem.A = sparse ([1 3 4], [1 2 4], [5 6+2i 7], 4, 4) ;
Problem.id = 7 ;
Problem.date = '2026' ;
Problem.author = 'Binsparse Developers' ;
Problem.ed = 'Binsparse Developers' ;
Problem.kind = 'test matrix' ;
Problem.Zeros = sparse ([2 4], [2 1], [1 1], 4, 4) ;
Problem.notes = ['first note  ' ; 'second note '] ;
Problem.b = [10 ; 20 ; 30 ; 40] ;
Problem.x = [1 2 ; 3 4 ; 5 6 ; 7 8] ;
Problem.aux = struct ;
Problem.aux.c = [1 ; 2 ; 3] ;
Problem.aux.D = [1 0 2 ; 3 4 5] ;
Problem.aux.S = sparse ([1 2], [2 3], [9 8], 3, 3) ;
Problem.aux.seq = {[11 ; 12] ; [21 ; 22]} ;
Problem.aux.note = ['hello ' ; 'there '] ;


%-------------------------------------------------------------------------------
% remove_workdir
%-------------------------------------------------------------------------------

function remove_workdir (work)

if (exist (work, 'dir'))
    rmdir (work, 's') ;
end


%-------------------------------------------------------------------------------
% assert_problem_equal
%-------------------------------------------------------------------------------

function assert_problem_equal (actual, expected, label)

actual_fields = sort (fieldnames (actual)) ;
expected_fields = sort (fieldnames (expected)) ;
assert (isequal (actual_fields, expected_fields), ...
    '%s field mismatch: actual {%s}, expected {%s}', label, ...
    strjoin (actual_fields, ', '), strjoin (expected_fields, ', ')) ;
for k = 1:numel (expected_fields)
    field = expected_fields{k} ;
    if (~isequaln (actual.(field), expected.(field)))
        if (strcmp (field, 'aux'))
            assert_aux_equal (actual.aux, expected.aux, label) ;
        end
        error ('ssbsp_test_ssread:Mismatch', ...
            '%s mismatch in Problem.%s (actual %s %s, expected %s %s)', ...
            label, field, class (actual.(field)), mat2str (size (actual.(field))), ...
            class (expected.(field)), mat2str (size (expected.(field)))) ;
    end
end


%-------------------------------------------------------------------------------
% assert_aux_equal
%-------------------------------------------------------------------------------

function assert_aux_equal (actual, expected, label)

actual_fields = sort (fieldnames (actual)) ;
expected_fields = sort (fieldnames (expected)) ;
assert (isequal (actual_fields, expected_fields), ...
    '%s aux field mismatch: actual {%s}, expected {%s}', label, ...
    strjoin (actual_fields, ', '), strjoin (expected_fields, ', ')) ;
for k = 1:numel (expected_fields)
    field = expected_fields{k} ;
    if (~isequaln (actual.(field), expected.(field)))
        error ('ssbsp_test_ssread:Mismatch', ...
            '%s mismatch in Problem.aux.%s (actual %s %s, expected %s %s)', ...
            label, field, class (actual.(field)), mat2str (size (actual.(field))), ...
            class (expected.(field)), mat2str (size (expected.(field)))) ;
    end
end
