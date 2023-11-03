////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_hash  /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief functions to deal with the hash, insert and find
 *
 * insert:
 *   gets key and value and put it into the already initailzed table with -1
 *   simple linear probing
 * find:
 *   find for the key and check if it has the correct value
 *
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"
// key*257 & mask
#define HASH_FUNCTION(key) (((key << 8) + (key)) & (hash_bits))

void paru_insert_hash(int64_t key, int64_t value, std::vector<int64_t> &colHash)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

#ifndef NDEBUG
    PRLEVEL(PR, ("%% Insert hash key=" LD " value=" LD " ", key, value));
    PRLEVEL(PR, ("size=" LD " \n", colHash.size()));
    PRLEVEL(PR, ("%% before insertion"));
    for (auto i : colHash) PRLEVEL(PR, (" " LD " ", i));
    PRLEVEL(PR, ("\n"));

#endif

    int64_t hash_bits = colHash.size() - 2;
    int64_t index = HASH_FUNCTION(key);

    int64_t loop_cnt = 0;
    while (colHash[index] != -1)
    {  // finding an empty spot
        index = (index + 1) & hash_bits;
        PRLEVEL(PR, ("index =" LD " colHash=" LD "\n", index, colHash[index]));
        loop_cnt++;
        ASSERT(loop_cnt < hash_bits);
    }
    colHash[index] = value;

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% hash_bits == %lx ", hash_bits));
    PRLEVEL(PR, ("%%"));
    for (auto i : colHash) PRLEVEL(PR, (" " LD " ", i));
    PRLEVEL(PR, ("\n"));
#endif
}

int64_t paru_find_hash(int64_t key, std::vector<int64_t> &colHash, int64_t *fcolList)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
#ifndef NDEBUG
    PRLEVEL(PR, ("%% find for hash key=" LD " \n", key));
#endif
    // lookup table
    if (colHash.back() == -1)
    {
        PRLEVEL(PR, ("%% LOOKUP key =" LD " colHash=" LD " \n", key, colHash[key]));
        return colHash[key];
    }

    int64_t hash_bits = colHash.size() - 2;
    int64_t index = HASH_FUNCTION(key);
    int64_t value = colHash[index];
    int64_t loop_cnt = 0;
    int64_t size = colHash.back();
    while (value != -1 && fcolList[value] != key)
    {
        index = (index + 1) & hash_bits;
        PRLEVEL(PR, ("%% index =" LD " \n", index));
        value = colHash[index];
        if (loop_cnt++ > log2(hash_bits))
        {  // take a long time in the hash;
            //  guarantees that find takes at most log time
            PRLEVEL(PR, ("%% binary search for hash\n"));
            value = paru_bin_srch(fcolList, 0, size - 1, key);
            break;
        }
    }

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%%"));
    for (auto i : colHash) PRLEVEL(PR, (" " LD " ", i));
    PRLEVEL(PR, ("\n"));
    PRLEVEL(PR, ("%% value is =" LD " \n", value));
    int64_t bsRes = paru_bin_srch(fcolList, 0, size - 1, key);
    PRLEVEL(PR, ("%% binSearch=" LD " \n", bsRes));
#endif
    return value;
}
