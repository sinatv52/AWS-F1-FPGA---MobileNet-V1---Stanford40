# MobileNet-V1 Implemetation on FPGA


## This Repo is goning to be Updated ...

This implementation is based on the upstream repo but with C++ wrapper (using xcl2.hpp and xcl2.cpp) to avoid "Not Releasing object" errors which is tuned for stanford40 dataset with 40 action classes.


- Used Techniques : Transfer Learning and Folding Batch Normalization 
- Network : MobileNet v1
- Dataset : Stanford40 (Action recognition)
- Target FPGA: AWS f1 (VU9P chip)
- Frameworks : Keras and tflite
- Software for FPGA : SDAccel v2018.3 in Centos 7.6
- Data Precision : Floating Point
- Input image format : Numpy Arrays extracted using Netron and  Reading using CNPY
