module load intel18
spack load miniconda3
source activate intelpy3
export CYTHON_HOME=$PWD
cd $CYTHON_HOME/1d
rm *.c
LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/2_loop
rm *.c
LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/4_loop/serial
rm *.c
LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries
cd $CYTHON_HOME/2d/4_loop/parallel
rm *.c
LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace
cp *.so $CYTHON_HOME/binaries