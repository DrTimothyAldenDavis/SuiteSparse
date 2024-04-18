////////////////////////////////////////////////////////////////////////////////
/////////////////////////// ParU_Version  //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief
 * The version number and date can also be obtained via compile-time constants
 * from ParU.h.  However, it is possible to compile a user application that
 * #includes one version of ParU.h and then links with another version of
 * the ParU library later on, so the version number and date may differ from
 * the compile-time constants.
 *
 *
 * @return  ParU_Info
 *
 *  @author Aznaveh
 */

#include "paru_internal.hpp"

ParU_Info ParU_Version (int ver [3], char date [128])
{
    // get version number and date
    ver[0] = PARU_VERSION_MAJOR ;
    ver[1] = PARU_VERSION_MINOR ;
    ver[2] = PARU_VERSION_UPDATE ;

    strncpy (date, PARU_DATE, 128) ;

    // make sure the date is null-terminated
    date[127] = '\0' ;
    return (PARU_SUCCESS) ;
}

