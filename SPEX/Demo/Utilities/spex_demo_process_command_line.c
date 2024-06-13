//------------------------------------------------------------------------------
// Demo/spex_process_command_line: processes command line for user specified options
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This processes the command line for user specified options */

#include "spex_demos.h"

SPEX_info spex_demo_process_command_line //processes the command line
(
    int64_t argc,           // number of command line arguments
    char *argv[],           // set of command line arguments
    SPEX_options option,    // struct containing the command options
    char **mat_name,        // Name of the matrix to be read in
    char **rhs_name,        // Name of the RHS vector to be read in
    int64_t *rat            // data type of output solution.
                            // 1: mpz, 2: double, 3: mpfr
)
{
    //SPEX_info ok;
    for (int64_t  i = 1; i < argc; i++)
    {
        char *arg = (char*) argv[i];
        if ( strcmp(arg,"help") == 0)
        {
            //SPEX_show_usage();
            return SPEX_INCORRECT_INPUT;
        }
        else if ( strcmp(arg, "q") == 0 || strcmp(arg,"col") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! There must be an argument between 1-3"
                    "following q\n 1: none, 2: Colamd, 3: AMD");
                return SPEX_INCORRECT_INPUT;
            }
            option->order = atoi(argv[i]);
            if (option->order < 1 || option->order > 3)
            {
                printf("\n****ERROR! Invalid column ordering"
                    "\nDefaulting to COLAMD\n\n");
                option->order = SPEX_COLAMD;
            }
        }
        else if ( strcmp(arg,"out2") == 0 || strcmp(arg, "o2") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! o2 or out2 must be followed by"
                    " 0 (print nothing) 1 (print err) or 2 (terse) \n");
                return SPEX_INCORRECT_INPUT;
            }
            else if (!atoi(argv[i]))
            {
                printf("\n****ERROR! o2 or out2 must be followed by"
                    " 0 (print nothing) 1 (print err) or 2 (terse) \n");
                return SPEX_INCORRECT_INPUT;
            }
            option->print_level = atoi(argv[i]);
        }
        else if ( strcmp(arg, "out") == 0 || strcmp(arg, "o") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! o or out must be followed by"
                    " 1 (rational) 2 (double) or 3 (variable precision) \n");
                return SPEX_INCORRECT_INPUT;
            }
            else if (!atoi(argv[i]))
            {
                printf("\n****ERROR! o or out must be followed by"
                    " 1 (rational) 2 (double) or 3 (variable precision) \n");
                return SPEX_INCORRECT_INPUT;
            }
            *rat = atoi(argv[i]);
            if (*rat < 1 || *rat > 3)
            {
                printf("\n\n****ERROR! Invalid output type!\n"
                   "Defaulting to rational");
                *rat = 1;
            }
            if (*rat == 3)
            {
                if (!argv[++i])
                {
                    printf("\n****ERROR! Precision must be specified\n");
                    return SPEX_INCORRECT_INPUT;
                }
                else if (!atoi(argv[i]))
                {
                    printf("\n****ERROR! Precision must be specified\n");
                    return SPEX_INCORRECT_INPUT;
                }
                option->prec = atoi(argv[i]);
                if (option->prec < 2)
                {
                    printf("\n\n****ERROR! Invalid precision. prec >= 2\n");
                    return SPEX_INCORRECT_INPUT;
                }
            }
        }
        else if ( strcmp(arg, "f") == 0 || strcmp(arg, "file") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! Matrix name must be entered\n");
                return SPEX_INCORRECT_INPUT;
            }
            *mat_name = argv[i];
            if (!argv[++i])
            {
                printf("\n****ERROR! Right hand side vector name must"
                    " be entered\n");
                return SPEX_INCORRECT_INPUT;
            }
            *rhs_name = argv[i];
        }
        else if ( strcmp(arg, "p") == 0 || strcmp(arg, "pivot") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! Pivoting scheme must be entered\n");
                printf("Options are: 0: Smallest, 1: Diagonal, 2: First nonzero,\n");
                printf("3: smallest with tolerance, 4: largest with tolerance, 5: largest");
                printf("\nDefaulting to smallest");
                option->pivot = SPEX_SMALLEST;
            }
            option->order = atoi(argv[i]);
        }
        else
        {
            printf("\n\n**ERROR! Unknown command line parameter: %s"
                    "\nIgnoring this parameter\n",arg);
            return SPEX_INCORRECT_INPUT;
        }
    }
    return SPEX_OK;
}
