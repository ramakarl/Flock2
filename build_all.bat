

REM ################################### Starting in Flock2 folder


REM ################################### Downloading or updating Libmin
cd ..
git clone https://github.com/ramakarl/libmin.git
cd libmin
git pull
cd ..


REM ################################### Building Libmin

cmake -S libmin -B build/libmin

path=%path%;"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin"

cd build/libmin
msbuild libmin.sln /p:Configuration=Debug
msbuild libmin.sln /p:Configuration=Release


REM ################################### Updating Flock2

cd ../../flock2
git pull
cd ..
cmake -S flock2 -B build/flock2 -DLIBMIN_INSTALL=build/libmin

cd build/flock2
msbuild flock2.sln /p:Configuration=Debug
msbuild flock2.sln /p:Configuration=Release
cd ../../flock2

REM ################################### DONE
REM ### Update and build complete.
REM ### To run flock2 debug, type:   ..\build\flock2\flock2d.exe 
REM ### To run flock2 release, type: ..\build\flock2\flock2.exe 
REM



