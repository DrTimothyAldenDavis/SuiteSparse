GraphBLAS/JITPackage:  package GraphBLAS source for the JIT 

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The use of this package is not required by the end user.  If you edit the
GraphBLAS source code itself, however, you must read the following
instructions.

This small stand-alone package compresses all the source files (*.c and *.h)
required by the JIT kernels into a single file: GB_JITpackage.c.  To ensure the
source files are up to date, the cmake build system for GraphBLAS always
constructs GB_JITpackage.c each time GraphBLAS is compiled.

When GraphBLAS starts, GrB_init checks the user source folder to ensure
~/.SuiteSparse/GrBx.y.z/src exists (where x.y.z is the current GraphBLAS
version number), and that it contains the GraphBLAS source code.  It does this
with a quick test: ~/.SuiteSparse/GrB.x.y.z/src/GraphBLAS.h must exist, and the
first line is checked to see if the version matches the GraphBLAS library
version.  If the file is not present or the version does not match, GrB_Init
uncompresses each file from its compressed form in GB_JITpackage.c, and writes
it to the user source folder.

If you edit the GraphBLAS source that goes into the GB_JITpackage.c, you must
delete your entire cache (simply delete the ~/.SuiteSparse/GrBx.y.z folder),
since these are updated only if the GraphBLAS version changes.  GrB_Init only
checks the first line of ~/.SuiteSparse/GrB.x.y.z/src/GraphBLAS.h.  It does not
check for any changes in the rest of the code.  If the src folder in the cache
changes then any prior compiled JIT kernels are invalidated.  It is also safest
to delete any GraphBLAS/PreJIT/* files; these will be recompiled properly if
the src cache files change, but any changes in other parts of GraphBLAS (the
JIT itself, in GraphBLAS/Source/*fy*c, in particular) can cause these kernels
to change.

A future version of GraphBLAS may do a more careful check (such as a CRC
checksum), so that this check would be automatic.  This would also guard
against a corrupted user cache.

