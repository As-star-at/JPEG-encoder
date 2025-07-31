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

## HOW TO USE THIS PROJECT IN GOOGLE COLAB

### INITIAL STEPS
- In a new notebook in Google Colab, paste the code in /src/main_code.cu
- At the top, wirte the following magic command: %%writefile jpeg.cu
- Save the image to be compressed in /content on Colab
  
### INSTALL REQUIRED DEPENDENCIES
This project requires updating package list and installation of OpenCV. Paste the following commands in a cell in Google Colab:<br><br>
```bash
!apt-get update
!apt-get install -y libopencv-dev pkg-config
```

### COMPILE CUDA CODE
Once the dependencies have been installed, the next step is to compile the code to generate an executable file. This is done by nvcc (NVIDIA CUDA Compiler) using the following commands. Paste them in a separate cell in Google Colab notebook:<br>
```bash
OPENCV_CFLAGS = !pkg-config --cflags opencv4
OPENCV_LIBS = !pkg-config --libs opencv4
!nvcc -o jpeg_converter jpeg.cu {OPENCV_CFLAGS[0]} {OPENCV_LIBS[0]} -arch=sm_75
```
Note that -arch=sm_75 flag is specific to the T4 GPU on Colab. It may need changes depending on the GPU being used.

### RUN THE ENCODER
The following command runs the jpeg_converter executable. It prints the step-by-step results of each stage of the pipeline, and produce a compressed JPEG output:<br>
```bash
!./jpeg_converter
```
