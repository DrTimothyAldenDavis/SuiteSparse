/* ========================================================================== */
/* === Include/Mongoose_Logger.hpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library, Copyright (C) 2017-2023,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * SPDX-License-Identifier: GPL-3.0-only
 * -------------------------------------------------------------------------- */

/**
 * Centralized debug and timing manager
 *
 * For debug and timing information to be displayed via stdout. This system
 * allows this information to be displayed (or not) without recompilation.
 * Timing inforation for different *portions of the library are also managed
 * here with a tic/toc pattern.
 */

// #pragma once
#ifndef MONGOOSE_LOGGER_HPP
#define MONGOOSE_LOGGER_HPP

#include <iostream>
#include <string>
#include <SuiteSparse_config.h>

#if !defined (SUITESPARSE_VERSION) || \
    (SUITESPARSE_VERSION < SUITESPARSE_VER_CODE(7,4))
#error "Mongoose requires SuiteSparse_config 7.4.0 or later"
#endif

// Default Logging Levels
#ifndef LOG_ERROR
#define LOG_ERROR 1
#endif

#ifndef LOG_WARN
#define LOG_WARN 0
#endif

#ifndef LOG_INFO
#define LOG_INFO 0
#endif

#ifndef LOG_TEST
#define LOG_TEST 0
#endif

// Main Logging Macros
#define LogError(msg)                                                          \
    do                                                                         \
    {                                                                          \
        if (LOG_ERROR)                                                         \
            (std::cout << __FILE__ << ":" << __LINE__ << ": " << msg);         \
    } while (0)
#define LogWarn(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_WARN)                                                          \
            (std::cout << __FILE__ << ":" << __LINE__ << ": " << msg);         \
    } while (0)
#define LogInfo(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_INFO)                                                          \
            (std::cout << msg);                                                \
    } while (0)
#define LogTest(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_TEST)                                                          \
            (std::cout << msg);                                                \
    } while (0)

namespace Mongoose
{

typedef enum DebugType
{
    None  = 0,
    Error = 1,
    Warn  = 2,
    Info  = 4,
    Test  = 8,
    All   = 15
} DebugType;

typedef enum TimingType
{
    MatchingTiming   = 0,
    CoarseningTiming = 1,
    RefinementTiming = 2,
    FMTiming         = 3,
    QPTiming         = 4,
    IOTiming         = 5
} TimingType;

class Logger
{
private:
    static int debugLevel;
    static bool timingOn;
    static double clocks[6];
    static float times[6];

public:
    static void tic(TimingType timingType);
    static void toc(TimingType timingType);
    static float getTime(TimingType timingType);
    static int getDebugLevel();
    static void setDebugLevel(int debugType);
    static void setTimingFlag(bool tFlag);
    static void printTimingInfo();
};

} // end namespace Mongoose

#endif
