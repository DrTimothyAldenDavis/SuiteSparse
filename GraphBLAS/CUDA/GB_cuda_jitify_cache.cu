//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_jitify_cache.cu
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
 * Copyright (c) 2019,2023 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//------------------------------------------------------------------------------

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#include "GB_cuda_jitify_cache.h"
#include "GraphBLAS_cuda.h"

extern "C"
{
    #include "GB.h"
    #include "GB_jitifyer.h"
}

namespace jit {

// Get the directory in home to use for storing the cache
    std::string get_user_home_cache_dir() {
        const char *path = GB_jitifyer_get_cache_path ( ) ;
        if (path == NULL)
        {
            return std::string ("") ;
        }
        else
        {
            return std::string (path) ;
        }
    }

GBJitCache::GBJitCache() { }

GBJitCache::~GBJitCache() { }


std::mutex GBJitCache::_kernel_cache_mutex;
std::mutex GBJitCache::_program_cache_mutex;

std::string GBJitCache::getFile(
    File_Desc &file_object )
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);

    // Macrofied version
    auto cached_file = getCachedFile( file_object, file_map );
    return *std::get<1>( cached_file ).get();
}

named_prog<jitify::experimental::Program> GBJitCache::getProgram(
    std::string const& prog_name, 
    std::string const& cuda_source,
    std::vector<std::string> const& given_headers,
    std::vector<std::string> const& given_options,
    jitify::experimental::file_callback_type file_callback)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);
//    printf(" jit_cache get program %s\n", prog_name.c_str());

    return getCached(prog_name, program_map, 
        [&](){
            return jitify::experimental::Program(cuda_source,
                                        given_headers,
                                        given_options,
                                        file_callback);
        }
    );
}

named_prog<jitify::experimental::KernelInstantiation> GBJitCache::getKernelInstantiation(
    std::string const& kern_name,
    named_prog<jitify::experimental::Program> const& named_program,
    std::vector<std::string> const& arguments)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_kernel_cache_mutex);

    std::string prog_name = std::get<0>(named_program);
    jitify::experimental::Program& program = *std::get<1>(named_program);

    // Make instance name e.g. "prog_binop.kernel_v_v_int_int_long int_Add"
    std::string kern_inst_name = kern_name;
    for ( auto&& arg : arguments ) kern_inst_name += '_' + arg;

    //printf(" got kernel instance %s\n",kern_inst_name.c_str());

    return getCached(kern_inst_name, kernel_inst_map, 
        [&](){return program.kernel(kern_name)
                            .instantiate(arguments);
        }
    );
}

// Another overload for getKernelInstantiation which might be useful to get
// kernel instantiations in one step
// ------------------------------------------------------------------------
/*
jitify::experimental::KernelInstantiation GBJitCache::getKernelInstantiation(
    std::string const& kern_name,
    std::string const& prog_name,
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    file_callback_type file_callback = nullptr)
{
    auto program = getProgram(prog_name,
                              cuda_source,
                              given_headers,
                              given_options,
                              file_callback);
    return getKernelInstantiation(kern_name, program);
}
*/

GBJitCache::cacheFile::cacheFile(std::string file_name)
 : _file_name{file_name}
{ }

GBJitCache::cacheFile::~cacheFile() { }

std::string GBJitCache::cacheFile::read_file()
{
    // Open file (duh)
    int fd = open ( _file_name.c_str(), O_RDWR );
    if ( fd == -1 ) {
        // TODO: connect errors to GrB_error result
//        printf(" failed to open cache file %s\n",_file_name.c_str());
        successful_read = false;
        return std::string();
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_read = false;
        return std::string();
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "rb" );

    // Get file length
    fseek( fp , 0L , SEEK_END);
    size_t file_size = ftell( fp );
    rewind( fp );

    // Allocate memory of file length size
    std::string content;
    content.resize(file_size);

    char *buffer = content.data();

    // Copy file into buffer
    if( fread(buffer, file_size, 1, fp) != 1 ) {
        //printf(" failed to read cache file %s\n",_file_name.c_str());
        successful_read = false;
        fclose(fp);
//        free(buffer); FIXME: Shouldn't need to free buffer since it's RAII
        return content; // FIXME: use unique_ptr here
    }

//    printf("about to close\n");
    fclose(fp);
    successful_read = true;
//    printf(" read cache file %s\n",_file_name.c_str());

    return content;
}

void GBJitCache::cacheFile::write(std::string content)
{
    // Open file and create if it doesn't exist, with access 0600
    int fd = open ( _file_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if ( fd == -1 ) {
        //printf(" failed to open cache file for write %s\n",_file_name.c_str());
        successful_write = false;
        return;
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_write = false;
        return;
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "wb" );

    // Copy string into file
    if( fwrite(content.c_str(), content.length(), 1, fp) != 1 ) {
        //printf(" failed to write cache file %s\n",_file_name.c_str());
        successful_write = false;
        fclose(fp);
        return;
    }
    fclose(fp);

    successful_write = true;
    //printf(" wrote cache file %s\n",_file_name.c_str());
    
    return;
}

} // namespace jit
