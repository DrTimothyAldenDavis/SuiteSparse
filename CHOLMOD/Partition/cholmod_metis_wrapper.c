//------------------------------------------------------------------------------
// CHOLMOD/Partition/cholmod_metis_wrapper: METIS functions embedded in CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Partition Module.  Copyright (C) 2005-2023, University of Florida.
// All Rights Reserved.  Author: Timothy A. Davis.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"
#undef ASSERT

#ifndef NPARTITION

//------------------------------------------------------------------------------
// redefine the METIS name space and include all METIS source files needed
//------------------------------------------------------------------------------

#include "cholmod_metis_wrapper.h"

#include "SuiteSparse_metis/GKlib/blas.c"
#include "SuiteSparse_metis/GKlib/error.c"
#include "SuiteSparse_metis/GKlib/evaluate.c"
#include "SuiteSparse_metis/GKlib/fkvkselect.c"
#include "SuiteSparse_metis/GKlib/mcore.c"
#include "SuiteSparse_metis/GKlib/memory.c"
#include "SuiteSparse_metis/GKlib/omp.c"
#include "SuiteSparse_metis/GKlib/random.c"
#include "SuiteSparse_metis/GKlib/sort.c"
#include "SuiteSparse_metis/GKlib/util.c"

// unused by CHOLMOD:
// #include "SuiteSparse_metis/GKlib/timers.c"
// replace the timer functions from timers.c:
double gk_WClockSeconds(void) { return (0) ; }
double gk_CPUSeconds(void) { return (0) ; }
// #include "SuiteSparse_metis/GKlib/fs.c"
// #include "SuiteSparse_metis/GKlib/getopt.c"
// #include "SuiteSparse_metis/GKlib/gkregex.c"
// #include "SuiteSparse_metis/GKlib/io.c"
// #include "SuiteSparse_metis/GKlib/pdb.c"
// #include "SuiteSparse_metis/GKlib/rw.c"
// #include "SuiteSparse_metis/GKlib/seq.c"
// #include "SuiteSparse_metis/GKlib/tokenizer.c"
// #include "SuiteSparse_metis/GKlib/string.c"
// #include "SuiteSparse_metis/GKlib/pqueue.c"
// #include "SuiteSparse_metis/GKlib/csr.c"
// #include "SuiteSparse_metis/GKlib/graph.c"
// #include "SuiteSparse_metis/GKlib/itemsets.c"
// #include "SuiteSparse_metis/GKlib/b64.c"
// #include "SuiteSparse_metis/GKlib/htable.c"

// for parmetis.c: replace abs with 64-bit version
#define abs SuiteSparse_metis_abs64

#include "SuiteSparse_metis/libmetis/auxapi.c"
#include "SuiteSparse_metis/libmetis/balance.c"
#include "SuiteSparse_metis/libmetis/bucketsort.c"
#include "SuiteSparse_metis/libmetis/coarsen.c"
#include "SuiteSparse_metis/libmetis/compress.c"
#include "SuiteSparse_metis/libmetis/contig.c"
#include "SuiteSparse_metis/libmetis/debug.c"
#include "SuiteSparse_metis/libmetis/fm.c"
#include "SuiteSparse_metis/libmetis/fortran.c"
#include "SuiteSparse_metis/libmetis/gklib.c"
#include "SuiteSparse_metis/libmetis/graph.c"
#include "SuiteSparse_metis/libmetis/initpart.c"
#include "SuiteSparse_metis/libmetis/kmetis.c"
#include "SuiteSparse_metis/libmetis/kwayfm.c"
#include "SuiteSparse_metis/libmetis/kwayrefine.c"
#include "SuiteSparse_metis/libmetis/mcutil.c"
#include "SuiteSparse_metis/libmetis/minconn.c"
#include "SuiteSparse_metis/libmetis/mincover.c"
#include "SuiteSparse_metis/libmetis/mmd.c"
#include "SuiteSparse_metis/libmetis/ometis.c"
#include "SuiteSparse_metis/libmetis/options.c"
#include "SuiteSparse_metis/libmetis/parmetis.c"
#include "SuiteSparse_metis/libmetis/pmetis.c"
#include "SuiteSparse_metis/libmetis/refine.c"
#include "SuiteSparse_metis/libmetis/separator.c"
#include "SuiteSparse_metis/libmetis/sfm.c"
#include "SuiteSparse_metis/libmetis/srefine.c"
#include "SuiteSparse_metis/libmetis/stat.c"
#include "SuiteSparse_metis/libmetis/timing.c"
#include "SuiteSparse_metis/libmetis/util.c"
#include "SuiteSparse_metis/libmetis/wspace.c"

// unused by CHOLMOD:
#ifndef NDEBUG
#include "SuiteSparse_metis/libmetis/checkgraph.c"
#endif
// #include "SuiteSparse_metis/libmetis/frename.c"
// #include "SuiteSparse_metis/libmetis/mesh.c"
// #include "SuiteSparse_metis/libmetis/meshpart.c"

#endif

