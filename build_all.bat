@echo off

echo ### FLOCK2
echo ### Rama Hoetzlein (c) 2023-2024
echo ### Flock2: A model for orientation-based social flocking
echo.
echo ### This script requires Git, Cmake and Visual Studio to be install before proceeding.
echo ### Also install CUDA Toolkit 10.2 or higher for GPU support.
echo. 
echo ### Stop and edit this .bat file to set your own Visual Studio path (if not using VS2019)
echo.

path=%path%;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin

echo ### Would you like to enable GPU build?

echo An NVIDIA GPU and CUDA Toolkit 10.2+ must be pre-installed for GPU support.
echo Select 'Y' to build for GPU. 'N' to build for CPU only.
setlocal
:PROMPT
SET /P USEGPU=Build Flock2 with GPU support (Y/[N])?
IF /I "%USEGPU%" NEQ "Y" GOTO GPU_OFF
:GPU_ON
endlocal
set CUDAOPT=-DBUILD_CUDA^=TRUE
GOTO PROCEED
:GPU_OFF
endlocal
set CUDAOPT=-DBUILD_CUDA^=FALSE
GOTO PROCEED

echo ### Proceeding with Flock2 install
:PROCEED

echo ### CUDA Option: %CUDAOPT%
echo.

echo ####### Folder structure created
echo #
echo # \Flock2    - flock2 repository
echo # \libmin    - libmin repository 
echo # \build
echo #   \flock2  - flock2 compiled
echo #   \libmin  - libmin compiled
echo #
echo # Assume this .bat file started in \Flock2 repository
echo.
echo ##################### Cloning or updating \libmin repository
echo.
cd ..
git clone https://github.com/ramakarl/libmin.git
cd libmin
git pull
cd ..

echo.
echo ##################### Compiling Libmin to \build\libmin
echo.

@echo on
cmake -S libmin -B build/libmin %CUDAOPT%

cd build/libmin
msbuild libmin.sln /p:Configuration=Debug
msbuild libmin.sln /p:Configuration=Release
@echo off

echo.
echo #################### Updating \Flock2 repository
echo.
cd ../../flock2
git pull
cd ..

echo.
echo ####### Compile Flock2 to \build\flock2
echo.

@echo on
cmake -S flock2 -B build/flock2 -DLIBMIN_INSTALL=build/libmin %CUDAOPT%

cd build/flock2
msbuild flock2.sln /p:Configuration=Debug
msbuild flock2.sln /p:Configuration=Release
@echo off


echo ######## DONE
echo ### Update and build complete.
echo ### Current directory is now \build\flock2
echo ### Type flock2.exe to run it!
echo.



