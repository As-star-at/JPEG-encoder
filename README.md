# IMAGE ENCODER USING PARALLEL PROGRAMMING
## INTRODUCTION
This project implements a custom JPEG image encoder, leveraging the power of parallel programming
using CUDA and OpenCV, with the core logic written in C++. The aim is to demonstrate the efficiency
and performance improvements achieved through parallelization as compared to the traditional sequential methods. 

## FEATURES
- Uses CUDA to parallelize portions of the JPEG encoder pipeline.
- Basic image handling tasks have been done through OpenCV.
- The language in which the code has been written in C++

## JPEG PIPELINE
The following steps have been implemented in the JPEG encoder pipeline, which takes an image input and produces a bitstream of compressed image characters as the output:
- Colour space conversion
- Chroma Downsampling
- Level shifting
- Discrete Cosine Transform
- Quantization
- Zig-zag scanning
- Run-length Encoding
- Huffman Coding
