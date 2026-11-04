mkdir -p build
cd build

cmake .. \
  -DCMAKE_CXX_FLAGS="-isystem $CONDA_PREFIX/include/eigen3" \
  -DBUILD_SHARED_LIBS=OFF \
  -DG2O_BUILD_APPS=OFF \
  -DG2O_USE_OPENGL=OFF \
  -DG2O_USE_CHOLMOD=OFF \
  -DG2O_USE_CSPARSE=OFF
make -j8
