export CYTHON_HOME=$PWD
cd $CYTHON_HOME/1d
rm *.c
python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/2_loop
rm *.c
python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/4_loop/serial
rm *.c
python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/4_loop/parallel
rm *.c
python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
