// ----------------------------------------------------------------------------
// SPEX/Tcov/simple_rand.h: a very simple random number generator
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

//  Since the simple_rand ( ) function is replicated from the POSIX.1-2001
//  standard, no copyright claim is intended for this specific file.  The
//  copyright statement above applies to all of SPEX, not this file.

#include <stdint.h>

#define SIMPLE_RAND_MAX 32767

/* return a random number between 0 and SIMPLE_RAND_MAX */
uint64_t simple_rand (void) ;

/* set the seed */
void simple_rand_seed (uint64_t seed) ;

/* get the seed */
uint64_t simple_rand_getseed (void) ;

/* return a random double between 0 and 1, inclusive */
double simple_rand_x ( void) ;

/* return a random uint64_t */
uint64_t simple_rand_i (void ) ;

