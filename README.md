# drone_controls
drone controls

# Requirements
control
slycot:
If using mac and pip install isn't working:
- double check gfortran is installed, then use ```Export FC=\path\to\gfortran\folder```
- if F2PY_EXECUTABLE not found, copy and paste [scikit-build's cmake](https://github.com/scikit-build/scikit-build/blob/master/skbuild/resources/cmake/FindF2PY.cmake) into the CMakeLists.txt of the downloaded slycot
