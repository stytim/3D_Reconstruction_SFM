# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project

# Include any dependencies generated for this target.
include CMakeFiles/gg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gg.dir/flags.make

CMakeFiles/gg.dir/test.cpp.o: CMakeFiles/gg.dir/flags.make
CMakeFiles/gg.dir/test.cpp.o: test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gg.dir/test.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gg.dir/test.cpp.o -c /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/test.cpp

CMakeFiles/gg.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gg.dir/test.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/test.cpp > CMakeFiles/gg.dir/test.cpp.i

CMakeFiles/gg.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gg.dir/test.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/test.cpp -o CMakeFiles/gg.dir/test.cpp.s

CMakeFiles/gg.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/gg.dir/test.cpp.o.requires

CMakeFiles/gg.dir/test.cpp.o.provides: CMakeFiles/gg.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/gg.dir/build.make CMakeFiles/gg.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/gg.dir/test.cpp.o.provides

CMakeFiles/gg.dir/test.cpp.o.provides.build: CMakeFiles/gg.dir/test.cpp.o


# Object files for target gg
gg_OBJECTS = \
"CMakeFiles/gg.dir/test.cpp.o"

# External object files for target gg
gg_EXTERNAL_OBJECTS =

gg: CMakeFiles/gg.dir/test.cpp.o
gg: CMakeFiles/gg.dir/build.make
gg: /Users/weiwei/anaconda3/lib/libopencv_stitching.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_superres.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_videostab.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_aruco.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_bgsegm.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_bioinspired.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_ccalib.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_dpm.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_face.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_freetype.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_fuzzy.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_hdf.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_img_hash.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_line_descriptor.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_optflow.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_reg.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_rgbd.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_saliency.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_stereo.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_structured_light.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_surface_matching.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_tracking.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_xfeatures2d.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_ximgproc.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_xobjdetect.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_xphoto.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_shape.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_photo.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_calib3d.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_phase_unwrapping.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_video.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_datasets.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_plot.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_text.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_dnn.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_features2d.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_flann.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_highgui.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_ml.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_videoio.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_imgcodecs.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_objdetect.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_imgproc.3.3.1.dylib
gg: /Users/weiwei/anaconda3/lib/libopencv_core.3.3.1.dylib
gg: CMakeFiles/gg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gg.dir/build: gg

.PHONY : CMakeFiles/gg.dir/build

CMakeFiles/gg.dir/requires: CMakeFiles/gg.dir/test.cpp.o.requires

.PHONY : CMakeFiles/gg.dir/requires

CMakeFiles/gg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gg.dir/clean

CMakeFiles/gg.dir/depend:
	cd /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project /Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/CMakeFiles/gg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gg.dir/depend

