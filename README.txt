PLANER: Tree Segmentation
Lynolan Moodley
CSC4002W
=========================

Project Content
===============
~ build/                CMake build files
~ data/                 DEM input data
~ docs/                 Program documentation
~ output/               Output files from the Program
~ CMakeLists.txt        CMake configuration file
~ DocConfig.txt         Doxygen configuration file
~ Evaluation.cpp
~ Evaluation.h
~ main.cpp
~ Preprocessing.cpp
~ Preprocessing.h
~ Segmentation.cpp
~ Segmentation.h
~ Watershed.cpp
~ Watershed.h

Installation Guide
==================
This project requires OpenCV

A. Opencv Installation
======================
i.  Detailed instructions can be found here: https://askubuntu.com/questions/1123955/install-opencv4-in-ubuntu-16
ii. Detailed instructions can be found here: https://towardsdatascience.com/how-to-install-opencv-and-extra-modules-from-source-using-cmake-and-then-set-it-up-in-your-pycharm-7e6ae25dbac5

1. Clone the opencv (main modules) and opencv_contrib (extra modules) repositiories from https://github.com/opencv
2. Create a folder called "build" in the main opencv directory.
3. Open CMake and point it to the source code directory and destination directory. Click "Configure" to configure the CMake parameters.
4. Browser tihe parameters displayed and select "OPENCV_EXTRA_MODULES_PATH". Remove any unnecessary modules, such as Java, Python and CUDA modules. Click "Generate".
5. Go to "build" folder. To create the makefile, run:
    $ cmake .
6. To compile OpenCV, run:
    $ make -j<numberOfCPUCores>
7. To install OpenCV, run:
    $ sudo make install

B. Program Compilation
======================
1. Edit the CMakeCache.txt file in the build directory of the project to point to the correct source code and build destination.
2. To generate the makefile, run:
    $ cmake ..
3. To compile, run:
    make

C. Program Execution
====================
1. Navigate to the build folder.
2. To execute the program, run:
    $ ./CSC4002W_Project -t(optional tag to enable test mode*) <fileNumber> <minimumPixelValue> <SLICSize> <localMaxWindowSize>

*The program is not designed to run files 34 to 36 in test mode.

D. File Numbers
===============
0.  contourHill
1.  contourHillEasy
2.  contourHillJoin
3.  contourHillJoinEasy
4.  contourHillJoinSpread
5.  contourHillSpread
6.  flat
7.  flatEasy
8.  flatSpread
9.  flatSpreadEasy
10. gentle
11. gentleEasy
12. gentleSpread
13. hills
14. hillsEasy
15. hillsSmooth
16. hillsSpread
17. steep
18. steepEasy
19. steepSpread
20. contourHillJoinSmall
21. contourHillJoinSmallEasy
22. contourHillSmall
23. contourHillSmallEasy
24. contourHillSmallSpread
25. flatSmall
26. flatSmallEasy
27. flatSmallSpread
28. gentleSmall
29. gentleSmallEasy
30. hillsSmall
31. hillsSmallEasy
32. steepSmall
33. steepSmallEasy
34. trim_easy
35. trim_medium
36. trim_hard