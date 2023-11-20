cmake_minimum_required(VERSION 3.16)

project(gpgpu) #Project name

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)


find_package(PkgConfig)
pkg_check_modules(GStreamer REQUIRED gstreamer-1.0 IMPORTED_TARGET)
pkg_check_modules(GStreamerVideo REQUIRED gstreamer-video-1.0 IMPORTED_TARGET)



add_library(gstcudafilter-base OBJECT src/gstcudafilter.c src/gstcudafilter.h src/logo.c)
target_link_libraries(gstcudafilter-base PkgConfig::GStreamerVideo)
set_target_properties(gstcudafilter-base PROPERTIES POSITION_INDEPENDENT_CODE ON)

# FILTRE en CUDA
add_library(gstcudafilter-cu SHARED src/filter_impl.h src/filter_impl.cu)
target_link_libraries(gstcudafilter-cu PkgConfig::GStreamerVideo gstcudafilter-base)

# FILTRE EN C++
add_library(gstcudafilter-cpp
             SHARED src/filter_impl.h src/filter_impl.cpp
             src/implem_cpp/background_update.cpp src/implem_cpp/background_update.hh
             src/implem_cpp/hysteresis.cpp src/implem_cpp/hysteresis.hh
             src/implem_cpp/remove_noise.cpp src/implem_cpp/remove_noise.hh
             src/implem_cpp/lab.cpp src/implem_cpp/lab.hh
             src/implem_cpp/mask.cpp src/implem_cpp/mask.hh
             src/implem_cpp/utils.hh
            )
target_link_libraries(gstcudafilter-cpp PkgConfig::GStreamerVideo gstcudafilter-base) 



add_executable(main src/main.cpp)
target_link_libraries(main PkgConfig::GStreamer)


