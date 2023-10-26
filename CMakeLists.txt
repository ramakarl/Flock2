cmake_minimum_required(VERSION 2.8)
set(PROJNAME Flock)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# LIBMIN Installed Path
#
set ( LIBMIN_PATH "d:/codes/build/libmin" CACHE STRING "Location of Libmin")
list( APPEND CMAKE_MODULE_PATH "${LIBMIN_PATH}/cmake" )
list( APPEND CMAKE_PREFIX_PATH "${LIBMIN_PATH}/cmake" )

#####################################################################################
# Include LIBMIN
#
find_package(Libmin QUIET)

if (NOT LIBMIN_FOUND)

  Message ( FATAL_ERROR "
  This project requires Libmin. Build Libmin first.
  Set the LIBMIN_PATH to the installed location (not the libmin source).
  Libmin also provides the cmakes used to build this app. 
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

_REQUIRE_MAIN()

set( USE_NVTX true CACHE BOOL "")

#--- symbols in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF" CACHE STRING "" FORCE)

###################################################################################
# OpenGL & GLEW
#
OPTION (BUILD_OPENGL "Build with OpenGL" ON)
if (BUILD_OPENGL)
    message ( STATUS "Searching for OpenGL and GLEW.." )
	find_package(OpenGL)
	add_definitions(-DUSE_OPENGL)  		# Use OpenGL
	IF (WIN32)
	     LIST(APPEND LIBRARIES_OPTIMIZED "opengl32.lib" )
	     LIST(APPEND LIBRARIES_DEBUG "opengl32.lib" )
	ENDIF()
	add_definitions(-DGLEW_STATIC)
	file(GLOB COMMON_SOURCE_FILES "${LIBMIN_GLEW_DIR}/glew.c" )   # Use GLEW (static)	
	message ( STATUS "  ---> Using GLEW: ${COMMON_SOURCE_FILES}" )
endif()

####################################################################################
# Find CUDA
#
if (NOT CUDA_TOOLKIT_ROOT_DIR) 
	set ( CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2" CACHE PATH "CUDA Toolkit path")
endif()
if ( NOT DEFINED CUDA_ARCH_BIN )
	message( "CUDA_ARCH_BIN not set. Using default. Set this to target GPU architecture.")
	SET( CUDA_ARCH_BIN "compute_50" CACHE STRING "CUDA Architecture target")
	SET( CUDA_ARCH_PTX "sm_50" CACHE STRING "CUDA Code target")
endif ()
message( STATUS "CUDA:") 
find_package(CUDA)

if ( CUDA_FOUND )
	message( STATUS "  --> Using package CUDA (ver ${CUDA_VERSION})") 
	add_definitions(-DUSE_CUDA)    
	include_directories(${CUDA_TOOLKIT_INCLUDE})
	LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
	LIST(APPEND LIBRARIES_DEBUG ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
	LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_TOOLKIT_INCLUDE} )    
	source_group(CUDA FILES ${CUDA_TOOLKIT_INCLUDE} ) 
	if (USE_NVTX)
		add_definitions(-DUSE_NVTX)
	endif()	
else()
	message ( FATAL_ERROR "  ---> Unable to find package CUDA")
endif()

####################################################################################
# Compile PTX Files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" *.cu *.cuh)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )

if ( USE_DEBUG_PTX )
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${LIBMIN_INC_DIR}" OPTIONS --ptx -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -G -g --use_fast_math --maxrregcount=32 )
else()
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${LIBMIN_INC_DIR}" OPTIONS --ptx -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -O3 --use_fast_math --maxrregcount=32 )
endif()

######################
# CMAKE_INSTALL_PREFIX -- path where library will be installed to

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   if ( WIN32 )
      get_filename_component ( _instpath "${CMAKE_CURRENT_BINARY_DIR}" REALPATH )
   else()
      get_filename_component ( _instpath "/usr/local/shapes" REALPATH )
   endif()
   set ( CMAKE_INSTALL_PREFIX ${_instpath} CACHE PATH "default install path" FORCE)   
endif()

#####################################################################################
# Asset Path
#
if ( NOT DEFINED ASSET_PATH ) 
   get_filename_component ( _assets "${BASE_DIRECTORY}/assets" REALPATH )
   set ( ASSET_PATH ${_assets} CACHE PATH "Full path to /assets" )
   add_definitions( -DASSET_PATH="${ASSET_PATH}/" )
endif()
file(GLOB GLSL_FILES ${ASSET_PATH}/*.glsl )
add_definitions(-DASSET_PATH="${ASSET_PATH}/")

#####################################################################################
# Executable
#
file(GLOB MAIN_FILES *.cpp *.c *.h )

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

_INSTALL_PTX ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )

# install ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

#####################################################################################
# Install Binaries
#
#
_DEFAULT_INSTALL_PATH()

# *NOTE*: file COPY is at cmake-time, not compile-time. Need to replace with add_custom_command -E copy_directory (my own _COPY)

file (COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets" DESTINATION ${EXECUTABLE_OUTPUT_PATH} )	# assets folder
_INSTALL ( FILES ${SHADERS} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/assets" )		# shaders
_INSTALL ( FILES ${PACKAGE_DLLS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )		# DLLs
install ( FILES $<TARGET_PDB_FILE:${PROJNAME}> DESTINATION ${EXECUTABLE_OUTPUT_PATH} OPTIONAL )		# PDB

install ( FILES ${INSTALL_LIST} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )		# exe, pdb

###########################
# Done
message ( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message ( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
message ( STATUS "EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH}" )




