//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_defaults: set CHOLMOD defaults
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// Sets all CHOLMOD parameters to their default values.

int CHOLMOD(defaults) (cholmod_common *Common)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;

    //--------------------------------------------------------------------------
    // set defaults
    //--------------------------------------------------------------------------

    Common->dbound = 0.0 ;              // bound D for LDL' (double case)
    Common->sbound = 0.0 ;              // bound D for LDL' (single case)
    Common->grow0 = 1.2 ;               // how to grow L for simplicial method
    Common->grow1 = 1.2 ;
    Common->grow2 = 5 ;
    Common->maxrank = 8 ;               // max rank for update/downdate

    Common->final_asis = TRUE ;         // leave L as-is
    Common->final_super = TRUE ;        // leave L supernodal
    Common->final_ll = FALSE ;          // leave factorization as LDL'
    Common->final_pack = TRUE ;         // pack columns when done
    Common->final_monotonic = TRUE ;    // sort columns when done
    Common->final_resymbol = FALSE ;    // do not resymbol when done

    Common->supernodal = CHOLMOD_AUTO ; // select supernodal automatically
    Common->supernodal_switch = 40 ;    // how to select super vs simpicial

    Common->prefer_zomplex = FALSE ;    // use complex, not zomplex
    Common->prefer_upper = TRUE ;       // sym case: use upper not lower
    Common->prefer_binary = FALSE ;     // use 1's when converting from pattern
    Common->quick_return_if_not_posdef = FALSE ;  // return early if not posdef

    Common->metis_memory = 0.0 ;        // metis memory control
    Common->metis_nswitch = 3000 ;
    Common->metis_dswitch = 0.66 ;

    Common->nrelax [0] = 4 ;            // supernodal relaxation parameters
    Common->nrelax [1] = 16 ;
    Common->nrelax [2] = 48 ;
    Common->zrelax [0] = 0.8 ;
    Common->zrelax [1] = 0.1 ;
    Common->zrelax [2] = 0.05 ;

    Common->print = 3 ;                 // print control
    Common->precise = FALSE ;           // print 5 digits

    //--------------------------------------------------------------------------
    // ordering methods
    //--------------------------------------------------------------------------

    Common->nmethods = 0 ;              // use default strategy
    Common->default_nesdis = FALSE ;    // use METIS not NESDIS
    Common->current = 0 ;               // method being evaluated
    Common->selected = EMPTY ;          // final method chosen
    Common->postorder = TRUE ;          // use weighted postordering

    // defaults for all methods (revised below)
    for (int i = 0 ; i <= CHOLMOD_MAXMETHODS ; i++)
    {
        Common->method [i].ordering = CHOLMOD_AMD ; // use AMD
        Common->method [i].fl = EMPTY ;             // no flop counts yet
        Common->method [i].lnz = EMPTY ;            // no lnz counts yet
        Common->method [i].prune_dense = 10.0 ;     // dense row/col parameter
        Common->method [i].prune_dense2 = -1 ;      // dense row/col parameter
        Common->method [i].aggressive = TRUE ;      // aggressive absorption
        Common->method [i].order_for_lu = FALSE ;   // order for chol, not lu
        Common->method [i].nd_small = 200 ;         // nesdis: small graph
        Common->method [i].nd_compress = TRUE ;     // nesdis: compression
        Common->method [i].nd_camd = 1 ;            // nesdis: use CAMD
        Common->method [i].nd_components = FALSE ;  // nesdis: use components
        Common->method [i].nd_oksep = 1.0 ;         // nesdis: for good sep
    }

    // define the 9 methods:
    // (0) given (skipped if no user permutation)
    // (1) amd
    // (2) metis
    // (3) nesdis with defaults
    // (4) natural
    // (5) nesdis: stop at subgraphs of 20000 nodes
    // (6) nesdis: stop at subgraphs of 4 nodes, do not use CAMD
    // (7) nesdis: no pruning on of dense rows/cols
    // (8) colamd

    Common->method [0].ordering = CHOLMOD_GIVEN ;
    Common->method [2].ordering = CHOLMOD_METIS ;
    Common->method [3].ordering = CHOLMOD_NESDIS ;
    Common->method [4].ordering = CHOLMOD_NATURAL ;
    Common->method [5].ordering = CHOLMOD_NESDIS ;
    Common->method [5].nd_small = 20000 ;
    Common->method [6].ordering = CHOLMOD_NESDIS ;
    Common->method [6].nd_small = 4 ;
    Common->method [6].nd_camd = 0 ;
    Common->method [7].ordering = CHOLMOD_NESDIS ;
    Common->method [7].prune_dense = -1. ;
    Common->method [8].ordering = CHOLMOD_COLAMD ;

    //--------------------------------------------------------------------------
    // GPU
    //--------------------------------------------------------------------------

    #if defined ( CHOLMOD_INT64 )
        // only use the GPU for the int64 version
        Common->useGPU = EMPTY ;
    #else
        Common->useGPU = 0 ;
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (TRUE) ;
}

