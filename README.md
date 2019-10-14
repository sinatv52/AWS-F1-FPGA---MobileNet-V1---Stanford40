# MobileNet-V1 Implemetation on FPGA


## This Repo is goning to be changed soon ...

This implementation is based on the upstream repo but with c++ wrapper (using xcl2.hpp and xcl2.cpp) to avoid "Not Releasing object" errors which is tuned for stanford40 dataset with 40 action classes.


- Used Techniques : Transfer Learning and Folding Batch Normalization 
- Network : MobileNet v1
- Dataset : Stanford40 (Action recognition)
- Target FPGA: AWS f1 (VU9P chip)
- Frameworks : Keras and tflite
- Software for FPGA : SDAccel
