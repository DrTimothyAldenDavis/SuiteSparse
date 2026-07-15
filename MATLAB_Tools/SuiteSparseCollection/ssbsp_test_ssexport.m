function ssbsp_test_ssexport
%SSBSP_TEST_SSEXPORT test opt-in Binsparse support in ssexport
%
% Uses the small HB/1138_bus problem to check BSP export and verification,
% backwards-compatible default formats, and format validation.

% SuiteSparseCollection, Copyright (c) 2006-2019, Timothy A Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

workdir = tempname ;
mkdir (workdir) ;
cleanup = onCleanup (@() remove_workdir (workdir)) ;

bspdir = fullfile (workdir, 'bsp') ;
ssexport (1, 'check', tempdir, {'BSP'}, bspdir) ;
assert (exist (fullfile (bspdir, 'BSP', 'HB', ...
    '1138_bus.tar.gz'), 'file') == 2, 'BSP archive was not created') ;

defaultdir = fullfile (workdir, 'default') ;
ssexport (1, '', tempdir, [ ], defaultdir) ;
assert (exist (fullfile (defaultdir, 'MM', 'HB', ...
    '1138_bus.tar.gz'), 'file') == 2, 'default MM archive was not created') ;
assert (exist (fullfile (defaultdir, 'RB', 'HB', ...
    '1138_bus.tar.gz'), 'file') == 2, 'default RB archive was not created') ;
assert (exist (fullfile (defaultdir, 'BSP'), 'dir') == 0, ...
    'default export unexpectedly created BSP output') ;

expect_error (@() ssexport (1, '', tempdir, {'BAD'}, workdir), ...
    'SuiteSparse:ssexport:InvalidFormat') ;
expect_error (@() ssexport (1, '', tempdir, {'BSP', 'bsp'}, workdir), ...
    'SuiteSparse:ssexport:DuplicateFormat') ;

fprintf ('ssbsp_test_ssexport: all tests passed\n') ;


%-------------------------------------------------------------------------------
% expect_error
%-------------------------------------------------------------------------------

function expect_error (f, identifier)

try
    f ( ) ;
catch me
    assert (strcmp (me.identifier, identifier), ...
        'expected %s, received %s', identifier, me.identifier) ;
    return
end
error ('expected error: %s', identifier) ;


%-------------------------------------------------------------------------------
% remove_workdir
%-------------------------------------------------------------------------------

function remove_workdir (workdir)

if (exist (workdir, 'dir'))
    rmdir (workdir, 's') ;
end
