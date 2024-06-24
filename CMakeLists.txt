cmake_minimum_required(VERSION 2.8...3.5)

set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "")

# CUDA policy. Use older FindCUDA (not using: cmake >3.27 first-class support for .cu files)
cmake_policy(SET CMP0146 OLD)

set(PROJNAME Flock2)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# LIBMIN Bootstrap
#
get_filename_component ( LIBMIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libmin/" REALPATH )
list( APPEND CMAKE_MODULE_PATH "${LIBMIN_ROOT}/cmake" )
list( APPEND CMAKE_PREFIX_PATH "${LIBMIN_ROOT}/cmake" )

#####################################################################################
# Include LIBMIN
#
find_package(Libmin QUIET)

if (NOT LIBMIN_FOUND)

  Message ( FATAL_ERROR "
  This project requires libmin. 
  Set LIBMIN_CMAKES to the libmin repository path for /libmin/cmake.
  " )

else()
  add_definitions(-DUSE_LIBMIN)  
  include_directories(${LIBMIN_INC_DIR})
  include_directories(${LIBRARIES_INC_DIR})  

  if (DEFINED ${BUILD_LIBMIN_STATIC})
    add_definitions(-DLIBMIN_STATIC) 
    file(GLOB LIBMIN_SRC "${LIBMIN_SRC_DIR}/*.cpp" )
    file(GLOB LIBMIN_INC "${LIBMIN_INC_DIR}/*.h" )
    LIST( APPEND LIBMIN_SOURCE_FILES ${LIBMIN_SRC} ${LIBMIN_INC} )
    message ( STATUS "  ---> Using LIBMIN (static)")
  else()    
    LIST( APPEND LIBRARIES_OPTIMIZED "${LIBMIN_LIB_DIR}/${LIBMIN_REL}")
    LIST( APPEND LIBRARIES_DEBUG "${LIBMIN_LIB_DIR}/${LIBMIN_DEBUG}")	     
    _EXPANDLIST( OUTPUT PACKAGE_DLLS SOURCE ${LIBMIN_LIB_DIR} FILES ${LIBMIN_DLLS} )
    message ( STATUS "  ---> Using LIBMIN")
  endif() 
endif()

#####################################################################################
# Options

set( USE_NVTX true CACHE BOOL "")

_REQUIRE_MAIN()

_REQUIRE_LIBEXT()       # for FFTW

_REQUIRE_GL()

_REQUIRE_GLEW()

OPTION ( BUILD_CUDA "Build CUDA for GPU." false )
if (BUILD_CUDA)
  _REQUIRE_CUDA( TRUE, "." )
endif()

#--- symbols in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF" CACHE STRING "" FORCE)

#####################################################################################
# FFTW
#
find_package(FFTW QUIET)

if (FFTW_FOUND)    
  add_definitions(-DUSE_FFTW)    
  list ( APPEND LIBRARIES_OPTIMIZED "${LIBEXT_REPO}/win64/libfftw3-3.lib" )
  list ( APPEND LIBRARIES_DEBUG "${LIBEXT_REPO}/win64/libfftw3-3.lib" )  
  include_directories( "${LIBEXT_REPO}/include" )  
  _EXPANDLIST( OUTPUT PACKAGE_DLLS SOURCE ${FFTW_LIB_DIR} FILES ${FFTW_DLL} )
  message ( STATUS "  ---> Using FFTW 3.3")  
else()
  message ( "WARNING: 
  FFTW not found. Frequency analysis will not be enabled. 
  Set cmake LIBEXT_REPO variable to a /libext which contains 
  FindFFTW.cmake and libfftw3-3.lib/dll/so.
  ")
endif()

#####################################################################################
# Asset Path
#
if ( NOT DEFINED ASSET_PATH ) 
   get_filename_component ( _assets "${CMAKE_CURRENT_SOURCE_DIR}/assets" REALPATH )
   set ( ASSET_PATH ${_assets} CACHE PATH "Full path to /assets" )   
endif()
file(GLOB GLSL_FILES ${ASSET_PATH}/*.glsl )
add_definitions(-DASSET_PATH="${ASSET_PATH}/")

#####################################################################################
# Executable
#
file(GLOB MAIN_FILES source/*.cpp source/*.c source/*.h )

unset ( ALL_SOURCE_FILES )

list( APPEND ALL_SOURCE_FILES ${MAIN_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${UTIL_SOURCE_FILES} )

if ( NOT DEFINED WIN32 )
    set(libdeps GL GLEW X11)
  LIST(APPEND LIBRARIES_OPTIMIZED ${libdeps})
  LIST(APPEND LIBRARIES_DEBUG ${libdeps})
ENDIF()
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")    

add_executable (${PROJNAME} ${ALL_SOURCE_FILES} ${CUDA_FILES} ${GLSL_FILES} )

set_property ( TARGET ${PROJNAME} APPEND PROPERTY DEPENDS )

#--- debug and release exe
set ( CMAKE_DEBUG_POSTFIX "d" CACHE STRING "" )
set_target_properties( ${PROJNAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

#####################################################################################
# Additional Libraries
#
_LINK ( PROJECT ${PROJNAME} OPT ${LIBRARIES_OPTIMIZED} DEBUG ${LIBRARIES_DEBUG} PLATFORM ${PLATFORM_LIBRARIES} )

#####################################################################################
# Windows specific
#
_MSVC_PROPERTIES()
source_group("Source Files" FILES ${MAIN_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group( CUDA FILES ${CUDA_FILES})

#####################################################################################
# Install Kernels
#

if (BUILD_CUDA)
   _INSTALL_PTX ( FILES ${CUDA_PTX_PATHS} DESTINATION ${CMAKE_INSTALL_PREFIX} OUTPUT INSTALL_LIST )
endif()

# install ( FILES ${CUDA_PTX_PATHS} DESTINATION ${CMAKE_INSTALL_PREFIX} )

#####################################################################################
# Install Binaries
#
#
_DEFAULT_INSTALL_PATH()

# *NOTE*: file COPY is at cmake-time, not compile-time. Need to replace with add_custom_command -E copy_directory (my own _COPY)

file (COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets" DESTINATION ${CMAKE_INSTALL_PREFIX} )	# assets folder
_INSTALL ( FILES ${GLEW_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX} )
_INSTALL ( FILES ${SHADERS} DESTINATION "${CMAKE_INSTALL_PREFIX}/assets" )		# shaders
_INSTALL ( FILES ${PACKAGE_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX} )		# DLLs
install ( FILES $<TARGET_PDB_FILE:${PROJNAME}> DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL )		# PDB

install ( FILES ${INSTALL_LIST} DESTINATION ${CMAKE_INSTALL_PREFIX} )		# exe, pdb

###########################
# Done
message ( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message ( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
message ( STATUS "------------------------------------")
message ( STATUS "Flock2 Install Location:  ${CMAKE_INSTALL_PREFIX}" )
message ( STATUS "------------------------------------")



