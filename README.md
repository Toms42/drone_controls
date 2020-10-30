# drone_controls
drone controls
http://tomscherlis.com/otw-portfolio/mpcdrone/
slides: https://docs.google.com/presentation/d/1c7nvPH4lnh2w3DKnrLayHfO5ZFBT8kSY3hoLUGLD9bQ/edit#slide=id.g8447ef9a51_1_0
paper: http://tomscherlis.com/wp-content/uploads/2020/05/drone_control_16899_final_report.pdf
video: https://www.youtube.com/watch?v=uN9TzCkSSKk&feature=emb_logo



# Requirements
control
slycot:
If using mac and pip install isn't working:
- double check gfortran is installed, then use ```Export FC=\path\to\gfortran\folder```
- if F2PY_EXECUTABLE not found, copy and paste [scikit-build's cmake](https://github.com/scikit-build/scikit-build/blob/master/skbuild/resources/cmake/FindF2PY.cmake) into the CMakeLists.txt of the downloaded slycot
