cmake_minimum_required(VERSION 2.8)
project( reconstruccion3D )
find_package(OpenCV REQUIRED )
find_package(PCL 1.8 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set (CMAKE_CXX_FLAGS "-fpermissive")

add_executable( Reconstruccion3D reconstruccion3D.cpp Distance.cpp Triangulation.cpp FindCameraMatrices.cpp MultiCameraDistance.cpp IDistance.h MultiCameraPnP.cpp Common.cpp IFeatureMatcher.h RichFeatureMatcher.cpp OFFeatureMatcher.cpp BundleAdjuster.cpp GPUSURFFeatureMatcher.cpp AbstractFeatureMatcher.cpp SfMUpdateListener.cpp)

target_link_libraries( Reconstruccion3D ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

