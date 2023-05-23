$directories = @("SuiteSparse_config", "Mongoose", "AMD", "BTF", "CAMD", "CCOLAMD", "COLAMD", "CHOLMOD", "CSparse", "CXSparse", "KLU", "LDL", "UMFPACK", "SPQR")

$blas_installation_path = "C:/OpenBLAS"
$working_directory = (get-location).path
$vs_generator = "Visual Studio 17 2022"
$arch = "x64"
$install_path = "./out/install"


$cmake = Get-Command cmake -ErrorAction SilentlyContinue

if ($null -eq $cmake) {
    Write-Output "CMake is not installed. Exit."
    exit
} 


foreach ($dir in $directories) {
    Write-Host "Processing module: $dir"
    
    cmake -S "./$dir" -B "./$dir/out/build" -G $vs_generator -A $arch -DCMAKE_PREFIX_PATH="$blas_installation_path;$working_directory/out/install"
    
    cmake --build "./$dir/out/build" --target ALL_BUILD --config Release
    
    cmake --install "./$dir/out/build" --prefix "$install_path"
}