% UMFPACK/Source2/create_source2.m: construct all UMFPACK/Source2 files

% UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+


clear all

UMFCH = {
   'umf_assemble', 'umf_blas3_update', 'umf_build_tuples', ...
   'umf_create_element', ...
   'umf_dump', 'umf_extend_front', 'umf_garbage_collection', ...
   'umf_get_memory', ...
   'umf_init_front', 'umf_kernel', 'umf_kernel_init', 'umf_kernel_wrapup', ...
   'umf_local_search', 'umf_lsolve', 'umf_ltsolve', 'umf_mem_alloc_element', ...
   'umf_mem_alloc_head_block', 'umf_mem_alloc_tail_block', ...
   'umf_mem_free_tail_block', 'umf_mem_init_memoryspace', ...
   'umf_report_vector', 'umf_row_search', 'umf_scale_column', ...
   'umf_set_stats', 'umf_solve', 'umf_symbolic_usage', 'umf_transpose', ...
   'umf_tuple_lengths', 'umf_usolve', 'umf_utsolve', 'umf_valid_numeric', ...
   'umf_valid_symbolic', 'umf_grow_front', 'umf_start_front', ...
   'umf_store_lu', 'umf_scale'
    }'

UMFINT = {
   'umf_analyze', 'umf_apply_order', 'umf_colamd', 'umf_free', 'umf_fsize', ...
   'umf_is_permutation', 'umf_malloc', 'umf_realloc', 'umf_report_perm', ...
   'umf_singletons', 'umf_cholmod'
}'

UMF_CREATED = {
   'umf_lhsolve', 'umf_uhsolve', 'umf_triplet_map_nox', ...
   'umf_triplet_nomap_nox', 'umf_triplet_map_x', ...
   'umf_assemble_fixq', 'umf_store_lu_drop'
    }'

UMF = [ UMF_CREATED ; UMFCH ]

UMFPACK = {
   'umfpack_col_to_triplet', 'umfpack_defaults', 'umfpack_free_numeric', ...
   'umfpack_free_symbolic', 'umfpack_get_numeric', 'umfpack_get_lunz', ...
   'umfpack_get_symbolic', 'umfpack_get_determinant', 'umfpack_numeric', ...
   'umfpack_qsymbolic', 'umfpack_report_control', 'umfpack_report_info', ...
   'umfpack_report_matrix', 'umfpack_report_numeric', 'umfpack_report_perm', ...
   'umfpack_report_status', 'umfpack_report_symbolic', ...
   'umfpack_report_triplet', ...
   'umfpack_report_vector', 'umfpack_solve', 'umfpack_symbolic', ...
   'umfpack_transpose', 'umfpack_triplet_to_col', 'umfpack_scale', ...
   'umfpack_load_numeric', 'umfpack_save_numeric', 'umfpack_copy_numeric', ...
   'umfpack_serialize_numeric', 'umfpack_deserialize_numeric', ...
   'umfpack_load_symbolic', 'umfpack_save_symbolic', ...
   'umfpack_copy_symbolic', ...
   'umfpack_serialize_symbolic', 'umfpack_deserialize_symbolic'
}'

UMFPACKW = { 'umfpack_wsolve' }'

UMFUSER = [UMFPACKW ; UMFPACK ]

GENERIC = { 'umfpack_timer', 'umfpack_tictoc' }'

%-------------------------------------------------------------------------------
% four versions of each file (di, dl, zi, zl):
%-------------------------------------------------------------------------------

kinds = { 'di', 'dl', 'zi', 'zl' } ;
defs1 = { 'DINT', 'DLONG', 'ZINT', 'ZLONG' } ;
whats = { 'double int32_t', 'double int64_t', ...
    'double complex int32_t', 'double complex int64_t' } ;
U = [UMFCH ; UMFPACK ]
for k = 1:length (U)
    file = U {k}
    for kk = 1:length (kinds)
        kind = kinds {kk} ;
        what = whats {kk} ;
        if (isequal (file (1:4), 'umf_'))
            % umf_*
            newfile = [file(1:4), kind, file(4:end), '.c' ] ;
        else
            % umfpack_*
            newfile = [file(1:8), kind, file(8:end), '.c'] ;
        end
        fprintf ('%s\n', newfile) ;
        f = fopen (newfile, 'w') ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
            newfile, what, file) ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
        fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '#define %s\n', defs1 {kk}) ;
        fprintf (f, '#include "%s.c"\n', file) ;
        fprintf (f, '\n') ;
        fclose (f) ;
    end
end

%-------------------------------------------------------------------------------
% special cases (four kinds each)
%-------------------------------------------------------------------------------

for kk = 1:length (kinds)
    kind = kinds {kk} ;
    what = whats {kk} ;

    file = 'umf_triplet' ;
    newfiles = { 'umf_%s_triplet_map_x.c', ...
                'umf_%s_triplet_map_nox.c', ...
                'umf_%s_triplet_nomap_x.c', ...
                'umf_%s_triplet_nomap_nox.c' } ;
    defs2 =  {
	'#define DO_MAP\n#define DO_VALUES\n', ...
	'#define DO_MAP\n', ...
	'#define DO_VALUES\n', ...
        '' } ;
    for i = 1:4
        newfile = sprintf (newfiles {i}, kind) ;
        fprintf ('%s\n', newfile) ;
        f = fopen (newfile, 'w') ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
            newfile, what, file) ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
        fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '#define %s\n', defs1 {kk}) ;
        fprintf (f, defs2 {i}) ;
        fprintf (f, '#include "%s.c"\n', file) ;
        fprintf (f, '\n') ;
        fclose (f) ;
    end

    file = 'umf_ltsolve' ;
    newfile = sprintf ('umf_%s_lhsolve.c', kind) ;
    fprintf ('%s\n', newfile) ;
    f = fopen (newfile, 'w') ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
        newfile, what, file) ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
    fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '#define %s\n', defs1 {kk}) ;
    fprintf (f, '#define CONJUGATE_SOLVE\n') ;
    fprintf (f, '#include "%s.c"\n', file) ;
    fprintf (f, '\n') ;
    fclose (f) ;

    file = 'umf_utsolve' ;
    newfile = sprintf ('umf_%s_uhsolve.c', kind) ;
    fprintf ('%s\n', newfile) ;
    f = fopen (newfile, 'w') ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
        newfile, what, file) ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
    fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '#define %s\n', defs1 {kk}) ;
    fprintf (f, '#define CONJUGATE_SOLVE\n') ;
    fprintf (f, '#include "%s.c"\n', file) ;
    fprintf (f, '\n') ;
    fclose (f) ;

    file = 'umf_assemble' ;
    newfile = sprintf ('umf_%s_assemble_fixq.c', kind) ;
    fprintf ('%s\n', newfile) ;
    f = fopen (newfile, 'w') ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
        newfile, what, file) ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
    fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '#define %s\n', defs1 {kk}) ;
    fprintf (f, '#define FIXQ\n') ;
    fprintf (f, '#include "%s.c"\n', file) ;
    fprintf (f, '\n') ;
    fclose (f) ;

    file = 'umf_store_lu' ;
    newfile = sprintf ('umf_%s_store_lu_drop.c', kind) ;
    fprintf ('%s\n', newfile) ;
    f = fopen (newfile, 'w') ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
        newfile, what, file) ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
    fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '#define %s\n', defs1 {kk}) ;
    fprintf (f, '#define DROP\n') ;
    fprintf (f, '#include "%s.c"\n', file) ;
    fprintf (f, '\n') ;
    fclose (f) ;

    file = 'umfpack_solve' ;
    newfile = sprintf ('umfpack_%s_wsolve.c', kind) ;
    fprintf ('%s\n', newfile) ;
    f = fopen (newfile, 'w') ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
        newfile, what, file) ;
    fprintf (f, '//------------------------------------------------------------------------------\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
    fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
    fprintf (f, '\n') ;
    fprintf (f, '#define %s\n', defs1 {kk}) ;
    fprintf (f, '#define WSOLVE\n') ;
    fprintf (f, '#include "%s.c"\n', file) ;
    fprintf (f, '\n') ;
    fclose (f) ;
end

%-------------------------------------------------------------------------------
% two versions of each file (i, l):
%-------------------------------------------------------------------------------

kinds = { 'i', 'l'} ;
defs1 = { 'DINT', 'DLONG' } ;
whats = { 'int32_t', 'int64_t' } ;
U = UMFINT
for k = 1:length (U)
    file = U {k} ;
    for kk = 1:length (kinds)
        kind = kinds {kk} ;
        what = whats {kk} ;
        if (isequal (file (1:4), 'umf_'))
            % umf_*
            newfile = [file(1:4), kind, file(4:end), '.c' ] ;
        else
            % umfpack_*
            newfile = [file(1:8), kind, file(8:end), '.c'] ;
        end
        fprintf ('%s\n', newfile) ;
        f = fopen (newfile, 'w') ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
            newfile, what, file) ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.\n') ;
        fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '#define %s\n', defs1 {kk}) ;
        fprintf (f, '#include "%s.c"\n', file) ;
        fprintf (f, '\n') ;
        fclose (f) ;
    end
end

%-------------------------------------------------------------------------------
% one versions of each file (gn):
%-------------------------------------------------------------------------------

kinds = { 'gn' } ;
whats = { 'generic' } ;
U = GENERIC
for k = 1:length (U)
    file = U {k} ;
    for kk = 1:length (kinds)
        kind = kinds {kk} ;
        what = whats {kk} ;
        if (isequal (file (1:4), 'umf_'))
            % umf_*
            newfile = [file(1:4), kind, file(4:end), '.c' ] ;
        else
            % umfpack_*
            newfile = [file(1:8), kind, file(8:end), '.c'] ;
        end
        fprintf ('%s\n', newfile) ;
        f = fopen (newfile, 'w') ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '// UMFPACK/Source2/%s:\n// %s version of %s\n', ...
            newfile, what, file) ;
        fprintf (f, '//------------------------------------------------------------------------------\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '// UMFPACK, Copyright (c) Timothy A. Davis, All Rights Reserved.\n') ;
        fprintf (f, '// SPDX-License-Identifier: GPL-2.0+\n') ;
        fprintf (f, '\n') ;
        fprintf (f, '#include "%s.c"\n', file) ;
        fprintf (f, '\n') ;
        fclose (f) ;
    end
end


