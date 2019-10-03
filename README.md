# How-to-Test-TVM-quantized-model-on-CUDA

git clone --recursive https://github.com/dmlc/tvm.git

vi .bashrc #Add the following two lines, and then reboot; NOTE! Set [YOUR_HOME_DIRECTORY] correctly!

#export TVM_HOME=/[YOUR_HOME_DIRECTORY]/tvm

#export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

#reboot

cd tvm

git checkout e22b5802a3e6c269d76e52428ca81cbd4b7d8304

git submodule update --init

mkdir build

cp cmake/config.cmake build

cd build

vi config.cmake #enable CUDA and LLVM

cmake ..

make -j4

cd ~

git clone --recursive https://github.com/vinx13/tvm-cuda-int8-benchmark.git

cd tvm-cuda-int8-benchmark

pip3 install mxnet-cu101mkl

python3 run_tvm.py --log_file logs/history_best_1080.log
