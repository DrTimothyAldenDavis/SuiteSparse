//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_common_tests: tests for cholmod_common
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

void common_tests (cholmod_common *cm)
{
    int cm_print_save = cm->print ;
    cm->print = 5 ;

    cm->status = CHOLMOD_GPU_PROBLEM ;
    CHOLMOD(print_common) ("cm:gpu", cm) ;
    cm->status = CHOLMOD_OK ;

    int save1 = cm->supernodal = -1 ;
    cm->supernodal = -1 ;
    CHOLMOD(print_common) ("cm:always_simplicial", cm) ;
    cm->supernodal = save1 ;

    save1 = cm->default_nesdis ;
    int save2 = cm->nmethods ;
    cm->nmethods = 0 ;
    cm->default_nesdis = true ;
    CHOLMOD(print_common) ("cm:default_nesdis", cm) ;
    cm->default_nesdis = save1 ;
    cm->nmethods = save2 ;

    save1 = cm->nmethods ;
    cm->nmethods = 2 ;
    save2 = cm->method [0].ordering ;
    int save3 = cm->method [1].ordering ;
    cm->method [0].ordering = CHOLMOD_NATURAL ;
    cm->method [1].ordering = CHOLMOD_METIS ;
    cm->method [2].fl = 32 ;
    cm->method [2].lnz = 8 ;
    CHOLMOD(print_common) ("cm:amd_backup", cm) ;
    cm->nmethods = save1 ;
    cm->method [0].ordering = save2 ;
    cm->method [1].ordering = save3 ;
    cm->method [2].fl = EMPTY ;
    cm->method [2].lnz = EMPTY ;

    save1 = cm->final_asis ;
    cm->final_asis = false ;
    save2 = cm->final_ll ;
    cm->final_ll = true ;
    CHOLMOD(print_common) ("cm:final_ll", cm) ;
    cm->final_asis = save1 ;
    cm->final_ll = save2 ;

    cm->print = cm_print_save ;
}

