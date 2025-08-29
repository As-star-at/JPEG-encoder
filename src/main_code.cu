%%writefile jpeg.cu
#include<iostream>
#include<opencv2/opencv.hpp>
#include<cuda_runtime.h>
#include<sys/time.h>
#include<cmath>
#include<vector>
#include<iomanip>

using namespace std;

__constant__ int zigzag_map[64][2] = {
    {0,0}, {0,1}, {1,0}, {2,0}, {1,1}, {0,2}, {0,3}, {1,2},
    {2,1}, {3,0}, {4,0}, {3,1}, {2,2}, {1,3}, {0,4}, {0,5},
    {1,4}, {2,3}, {3,2}, {4,1}, {5,0}, {6,0}, {5,1}, {4,2},
    {3,3}, {2,4}, {1,5}, {0,6}, {0,7}, {1,6}, {2,5}, {3,4},
    {4,3}, {5,2}, {6,1}, {7,0}, {7,1}, {6,2}, {5,3}, {4,4},
    {3,5}, {2,6}, {1,7}, {2,7}, {3,6}, {4,5}, {5,4}, {6,3},
    {7,2}, {7,3}, {6,4}, {5,5}, {4,6}, {3,7}, {4,7}, {5,6},
    {6,5}, {7,4}, {7,5}, {6,6}, {5,7}, {6,7}, {7,6}, {7,7}
};

typedef struct{
    //This struct represents a pixel in a coloured image with RGB color space.
    //It has 3 values: red, green and blue
    //Need to make r, g, b unsigned char so that they take values only from 0-255
    unsigned char r, g, b;
} rgb;
typedef struct{
    //This struct represents a pixel in a coloured img with YCbCr space.
    unsigned char y, cr, cb;
} ycbcr;

typedef struct encoded_symbol{
    int run_len; // # of preceding zeroes (always 0 for the DC coefficient as nothing precedes it)
    int size; // # of bits needed to represent value
    int value; // The non-zero term that ends a string of zeroes.
};

void check(cudaError_t err){
    if(err != cudaSuccess){
        cout<<"Error: "<<cudaGetErrorString(err);
    }
    return;
}

//Function to measure time for CPU execution only.
double cpu_sec(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double(tp.tv_sec + tp.tv_usec*1.0E-06));
}

//Device side function for colour space conversion from RGB to YCrCb.
__global__ void rgb_to_ycbcr(rgb* d_rgb, ycbcr* d_ycbcr, int height, int width){
    int i= blockDim.y * blockIdx.y + threadIdx.y; //row
    int j= blockDim.x * blockIdx.x + threadIdx.x; //col
    if(i>= height || j>= width){return;}
    int index= i*width + j;
    rgb rgb_pixel= d_rgb[index];
    float red= (float)rgb_pixel.r;
    float green= (float)rgb_pixel.g;
    float blue= (float)rgb_pixel.b;

    ycbcr ycbcr_pixel;
    ycbcr_pixel.y= static_cast<unsigned char> (0.299f * red + 0.587f * green + 0.114f * blue); 
    ycbcr_pixel.cr= static_cast<unsigned char>(0.5f * red - 0.418688f * green - 0.081312f * blue + 128.0f);
    ycbcr_pixel.cb= static_cast<unsigned char>(-0.168736f * red - 0.331264f * green + 0.5f * blue + 128.0f);
    d_ycbcr[index]= ycbcr_pixel;
}

//Device side kernel for downsamoling
__global__ void downsample(ycbcr* d_ycbcr, ycbcr* d_ycbcr_downsampled, int height, int width){
    int i= blockDim.y * blockIdx.y + threadIdx.y;
    int j= blockDim.x * blockIdx.x + threadIdx.x;
    if(i>= height || j>= width) return;
    int index= i*width + j;

    //retain the entire content of Y component of the pixel
    d_ycbcr_downsampled[index].y= d_ycbcr[index].y;

    //Averaging method: Divide the matrix into blocks of 2*2. Calculate the avg of all values, and replace all the elements in this block by the calculated avg value.
    int count = 0; //to handle edge cases
    float cr_sum=0, cb_sum=0;

    //The below two conditions are for determining the top left corner of a 2*2 block.
    int row_start= (i/2) * 2;
    int col_start= (j/2) * 2;

    for(int r=0; r<2; r++){
        for(int c=0; c<2; c++){
            int cur_row= row_start + r;
            int cur_col= col_start + c;
            if(cur_row < height && cur_col < width){
                //this condition is for bounds check. If there are odd number of rows/columns in the matrix, this condition will handle the edge cases.
                int ind= cur_row * width + cur_col;
                cr_sum += d_ycbcr[ind].cr;
                cb_sum += d_ycbcr[ind].cb;
                count++;
            }
        }
    }
    unsigned char avg_cb= (count>0)? static_cast<unsigned char>(cb_sum/count) : 0;
    unsigned char avg_cr= (count>0)? static_cast<unsigned char>(cr_sum/count) : 0;
    //finally assign values
    d_ycbcr_downsampled[index].cr= avg_cr;
    d_ycbcr_downsampled[index].cb= avg_cb;
}

//Helper function for the DCT kernel. Runs only on the GPU.
__device__ float alpha(int a){
    if(a==0){return (float)(1.0f/sqrtf(2.0f));}
    else return 1.0f;
}

//DCT kernel
__global__ void dct(const ycbcr* d_ycbcr_downsampled, float* d_dct_y, float* d_dct_cr, float* d_dct_cb, int height, int width){
    const float PI = 3.14159265358979323846f;
    int block_start_y= blockDim.y * blockIdx.y;
    int block_start_x= blockDim.x * blockIdx.x;
    if(block_start_x >= width || block_start_y >= height){
        return;
    }

    __shared__ float level_y[8][8];
    __shared__ float level_cr[8][8];
    __shared__ float level_cb[8][8];

    int local_x= threadIdx.x;
    int local_y= threadIdx.y;
    int global_x= block_start_x + local_x;
    int global_y= block_start_y + local_y;

    //Level shifting
    if(global_x < width && global_y < height){
        level_y[local_y][local_x]= (float)(d_ycbcr_downsampled[global_y*width + global_x].y) -128.0f;
        level_cr[local_y][local_x]= (float)(d_ycbcr_downsampled[global_y*width + global_x].cr) -128.0f;
        level_cb[local_y][local_x]= (float)(d_ycbcr_downsampled[global_y*width + global_x].cb) -128.0f;
    }
    else{
        level_y[local_y][local_x]= 0.0f;
        level_cr[local_y][local_x]= 0.0f;
        level_cb[local_y][local_x]= 0.0f;
    }
    __syncthreads();

    float cos_sum_y=0.0f;
    float cos_sum_cr= 0.0f;
    float cos_sum_cb= 0.0f;

    for(int x=0; x<8; x++){
        for(int y=0; y<8; y++){
            cos_sum_y +=   level_y[x][y] * cosf(((2*x + 1) * PI * local_x)/16) * cosf(((2*y + 1) * PI * local_y)/16);
            cos_sum_cr += level_cr[x][y] * cosf(((2*x + 1) * PI * local_x)/16) * cosf(((2*y + 1) * PI * local_y)/16);
            cos_sum_cb += level_cb[x][y] * cosf(((2*x + 1) * PI * local_x)/16) * cosf(((2*y + 1) * PI * local_y)/16);
        }
    }
    float normalize= 0.25f;
    float au= alpha(local_y);
    float av= alpha(local_x);
    if(global_y < height && global_x < width){
        int index= (global_y * width + global_x);
        d_dct_y[index]= normalize * au * av * cos_sum_y;
        d_dct_cr[index]= normalize * au * av * cos_sum_cr;
        d_dct_cb[index]= normalize * au * av * cos_sum_cb;
    }
}

//Helper function for sequential DCT
float alpha_cpu(int a){
    if(a==0){return (float)(1.0f/sqrtf(2.0f));}
    else return 1.0f;
}

//Function used for sequentially executing DCT using OpenCV built-in functions
void sequential_dct_opencv(const cv::Mat& h_ycbcr_downsampled_mat, float* h_dct_y, float* h_dct_cr, float* h_dct_cb, int height, int width) {
    // Ensure the input matrix is 3-channel for YCbCr
    if (h_ycbcr_downsampled_mat.channels() != 3) {
        std::cerr << "Error: Input YCbCr matrix for sequential_dct_opencv must have 3 channels (Y, Cb, Cr)." << std::endl;
        return;
    }

    for (int block_start_y = 0; block_start_y < height; block_start_y += 8) {
        for (int block_start_x = 0; block_start_x < width; block_start_x += 8) {
            cv::Mat block_y(8, 8, CV_32F);
            cv::Mat block_cr(8, 8, CV_32F);
            cv::Mat block_cb(8, 8, CV_32F);

            // Populate 8x8 block and perform level shift
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    int global_y = block_start_y + r;
                    int global_x = block_start_x + c;

                    // Access pixels from the cv::Mat input
                    if (global_y < height && global_x < width) {
                        cv::Vec3b ycbcr_pixel = h_ycbcr_downsampled_mat.at<cv::Vec3b>(global_y, global_x);
                        block_y.at<float>(r, c) = (float)ycbcr_pixel[0] - 128.0f; // Assuming Y is channel 0
                        block_cr.at<float>(r, c) = (float)ycbcr_pixel[1] - 128.0f; // Assuming Cb is channel 1
                        block_cb.at<float>(r, c) = (float)ycbcr_pixel[2] - 128.0f; // Assuming Cr is channel 2
                    } else {
                        // Pad with 0 for blocks at the image edges
                        block_y.at<float>(r, c) = 0.0f;
                        block_cr.at<float>(r, c) = 0.0f;
                        block_cb.at<float>(r, c) = 0.0f;
                    }
                }
            }

            // Perform 2D DCT using OpenCV's cv::dct()
            cv::Mat dct_block_y;
            cv::Mat dct_block_cr;
            cv::Mat dct_block_cb;

            cv::dct(block_y, dct_block_y);
            cv::dct(block_cr, dct_block_cr);
            cv::dct(block_cb, dct_block_cb);

            // Copy results back to your flat arrays h_dct_y, h_dct_cr, h_dct_cb
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    int global_idx_u = block_start_y + r;
                    int global_idx_v = block_start_x + c;

                    if (global_idx_u < height && global_idx_v < width) {
                        int index = global_idx_u * width + global_idx_v;
                        h_dct_y[index] = dct_block_y.at<float>(r, c);
                        h_dct_cr[index] = dct_block_cr.at<float>(r, c);
                        h_dct_cb[index] = dct_block_cb.at<float>(r, c);
                    }
                }
            }
        }
    }
}

//Normal sequential DCT function
void sequential_dct(const ycbcr* h_ycbcr_downsampled, float* h_dct_y, float* h_dct_cr, float* h_dct_cb, int height, int width){
    const float PI = 3.14159265358979323846f;
    const float normalize = 0.25f;

    for (int block_start_y = 0; block_start_y < height; block_start_y += 8) {
        for (int block_start_x = 0; block_start_x < width; block_start_x += 8) {
            float level_y[8][8];
            float level_cr[8][8];
            float level_cb[8][8];

            // Populate 8x8 block and perform level shift
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    int global_y = block_start_y + r;
                    int global_x = block_start_x + c;

                    if (global_y < height && global_x < width) {
                        level_y[r][c] = (float)(h_ycbcr_downsampled[global_y * width + global_x].y) - 128.0f;
                        level_cr[r][c] = (float)(h_ycbcr_downsampled[global_y * width + global_x].cr) - 128.0f;
                        level_cb[r][c] = (float)(h_ycbcr_downsampled[global_y * width + global_x].cb) - 128.0f;
                    } else {
                        // Pad with 0 for blocks at the image edges
                        level_y[r][c] = 0.0f;
                        level_cr[r][c] = 0.0f;
                        level_cb[r][c] = 0.0f;
                    }
                }
            }

            // Perform 2D DCT for each component
            for (int u = 0; u < 8; ++u) { // Frequency domain rows
                for (int v = 0; v < 8; ++v) { // Frequency domain columns
                    float cos_sum_y = 0.0f;
                    float cos_sum_cr = 0.0f;
                    float cos_sum_cb = 0.0f;

                    for (int x = 0; x < 8; ++x) { // Spatial domain rows
                        for (int y = 0; y < 8; ++y) { // Spatial domain columns
                            cos_sum_y += level_y[x][y] * cosf(((2 * x + 1) * PI * u) / 16) * cosf(((2 * y + 1) * PI * v) / 16);
                            cos_sum_cr += level_cr[x][y] * cosf(((2 * x + 1) * PI * u) / 16) * cosf(((2 * y + 1) * PI * v) / 16);
                            cos_sum_cb += level_cb[x][y] * cosf(((2 * x + 1) * PI * u) / 16) * cosf(((2 * y + 1) * PI * v) / 16);
                        }
                    }

                    float au = alpha_cpu(u);
                    float av = alpha_cpu(v);

                    int global_idx_u = block_start_y + u;
                    int global_idx_v = block_start_x + v;

                    if (global_idx_u < height && global_idx_v < width) {
                        int index = global_idx_u * width + global_idx_v;
                        h_dct_y[index] = normalize * au * av * cos_sum_y;
                        h_dct_cr[index] = normalize * au * av * cos_sum_cr;
                        h_dct_cb[index] = normalize * au * av * cos_sum_cb;
                    }
                }
            }
        }
    }
}

//Quantization kernel
__global__ void quantization(int* q_y, int* q_cr, int* q_cb, float* d_dct_y, float* d_dct_cr, float* d_dct_cb, int height, int width){
    //This table is for Y components
    const unsigned char qtable_Y[64] = {
        16,11,10,16,24,40,51,61,
        12,12,14,19,26,58,60,55,
        14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62,
        18,22,37,56,68,109,103,77,
        24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101,
        72,92,95,98,112,100,103,99
    };

    //This is for the Cr and Cb components
    const unsigned char qtable_C[64] = {
        17,18,24,47,99,99,99,99,
        18,21,26,66,99,99,99,99,
        24,26,56,99,99,99,99,99,
        47,66,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99
    };
    
    int local_x= threadIdx.x;
    int local_y= threadIdx.y;
    int block_start_x= blockDim.x * blockIdx.x;
    int block_start_y= blockDim.y * blockIdx.y;
    int global_x= block_start_x + local_x;
    int global_y= block_start_y + local_y;
    if(global_x < width && global_y < height){
        float y_numerator= d_dct_y[global_y * width + global_x];
        float cr_numerator= d_dct_cr[global_y * width + global_x];
        float cb_numerator= d_dct_cb[global_y * width + global_x];
        unsigned char y_denominator = qtable_Y[local_y*8 + local_x];
        unsigned char c_denominator= qtable_C[local_y*8 + local_x];

        q_y[global_y * width + global_x] = roundf(y_numerator/y_denominator);
        q_cr[global_y * width + global_x] = roundf(cr_numerator/c_denominator);
        q_cb[global_y * width + global_x] = roundf(cb_numerator/c_denominator);
    }
}

//Encoding- zig zag scan
__global__ void zig_zag_scan(int* zz_scanned_y, int* zz_scanned_cr, int* zz_scanned_cb, int* q_y, int* q_cr, int* q_cb, int height, int width){
    int block_start_x= blockIdx.x * blockDim.x;
    int block_start_y= blockIdx.y * blockDim.y;
    if(block_start_x >= width || block_start_y >= height) return;

    __shared__ int zz_y[8][8];
    __shared__ int zz_cr[8][8];
    __shared__ int zz_cb[8][8];
    int local_x= threadIdx.x;
    int local_y= threadIdx.y;
    int global_x= block_start_x + local_x;
    int global_y= block_start_y + local_y;
    
    if(global_x >= width || global_y >= height){
        zz_y[local_y][local_x]=0;
        zz_cr[local_y][local_x]=0;
        zz_cb[local_y][local_x]=0;
    }
    else{
        zz_y[local_y][local_x]= q_y[global_y * width + global_x];
        zz_cr[local_y][local_x]= q_cr[global_y * width + global_x];
        zz_cb[local_y][local_x]= q_cb[global_y * width + global_x];
    }
    __syncthreads();

    int local_thread_id= local_y * 8 + local_x;
    int a= zigzag_map[local_thread_id][0];
    int b= zigzag_map[local_thread_id][1];

    //to calculate how many 8x8 blocks fit horizontally across the image's width
    int total_blocks_per_row = (width + 7) / 8;

    //to calculate the base index for the current block in the 1D output array: output_base_idx is the 
    int output_base_idx = (blockIdx.y * total_blocks_per_row + blockIdx.x) * 64;

    zz_scanned_y[output_base_idx + local_thread_id]= zz_y[a][b];
    zz_scanned_cr[output_base_idx + local_thread_id]= zz_cr[a][b];
    zz_scanned_cb[output_base_idx + local_thread_id]= zz_cb[a][b];
}

//For RLE to return the size of code for a non-zero value. Based on the binary values of numbers.
int get_code_bits(int value){
    if (value == 0) return 0; // Category 0 for value 0
    int abs_val = std::abs(value);
    int size = 0;
    if (abs_val == 1) size = 1;
    else if (abs_val <= 3) size = 2;
    else if (abs_val <= 7) size = 3;
    else if (abs_val <= 15) size = 4;
    else if (abs_val <= 31) size = 5;
    else if (abs_val <= 63) size = 6;
    else if (abs_val <= 127) size = 7;
    else if (abs_val <= 255) size = 8;
    else if (abs_val <= 511) size = 9;
    else if (abs_val <= 1023) size = 10;
    else if (abs_val <= 2047) size = 11;
    return size;
}

//Function to build the huffman table
std::map<unsigned char, std::pair<unsigned short, unsigned char>> build_huffman_table(
    const unsigned char* bits, const unsigned char* huffval) {
    std::map<unsigned char, std::pair<unsigned short, unsigned char>> huff_table;
    unsigned short code = 0;
    int huffval_idx = 0;

    for (int len = 1; len <= 16; ++len) { // Max 16 bits for Huffman codes
        for (int i = 0; i < bits[len]; ++i) {
            unsigned char symbol = huffval[huffval_idx++];
            huff_table[symbol] = {code, static_cast<unsigned char>(len)};
            code++;
        }
        code <<= 1; // Shift left for the next length
    }
    return huff_table;
}

//Function to get the additional bits for a value
unsigned short get_value_bits(int value, int size) {
    if (value == 0) return 0; // Should not happen for non-zero values
    if (value > 0) {
        return static_cast<unsigned short>(value);
    } else { // value < 0
        // For negative values, the size bits are the one's complement of the absolute value
        return static_cast<unsigned short>(~std::abs(value) & ((1 << size) - 1));
    }
}

//bitstream writer class- used for Huffman coding
class BitStreamWriter {
public:
    std::vector<unsigned char> data;
    unsigned int bit_buffer;
    int bits_in_buffer;

    BitStreamWriter() : bit_buffer(0), bits_in_buffer(0) {}

    void write_bits(unsigned short code, unsigned char length) {
        bit_buffer |= (code << (32 - bits_in_buffer - length));
        bits_in_buffer += length;

        while (bits_in_buffer >= 8) {
            unsigned char byte = static_cast<unsigned char>((bit_buffer >> 24) & 0xFF);
            data.push_back(byte);
            if (byte == 0xFF) { // JPEG Byte Stuffing: if 0xFF, insert 0x00
                data.push_back(0x00);
            }
            bit_buffer <<= 8;
            bits_in_buffer -= 8;
        }
    }

    void flush_bits() {
        if (bits_in_buffer > 0) {
            unsigned char byte = static_cast<unsigned char>((bit_buffer >> (32 - 8)) & 0xFF); // Corrected shift for remaining bits
            data.push_back(byte);
            if (byte == 0xFF) { // JPEG Byte Stuffing for the last byte
                data.push_back(0x00);
            }
        }
        bit_buffer = 0;
        bits_in_buffer = 0;
    }
};

int main(){
    string img_path;
    cout<<"Enter the path to image: ";
    cin>> img_path;

    string original_img= img_path;

    //img is basically the matrix on which our image is stored. No need to allocate extra memory for host input data.
    cv::Mat img= cv::imread(original_img);
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB); //by default image will be read in BGR format. Need to convert it to RGB format.
    if(img.empty()){cout<<"Couldn't open image\n"; return 1;}
    if(img.type() != CV_8UC3){cout<<"Image must be 8-bit, 3 channel type\n"; return 1;}    

    //Events creation for CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int height= img_rgb.rows;
    int width= img_rgb.cols;
    size_t rgb_bytes= height * width * sizeof(rgb);
    size_t ycbcr_bytes= height * width * sizeof(ycbcr);

    //allocate memory for device
    rgb* d_rgb;
    check(cudaMalloc((void**) &d_rgb, rgb_bytes));
    //Transfer data from host to device
    check(cudaMemcpy(d_rgb, img_rgb.data, rgb_bytes, cudaMemcpyHostToDevice));

    //Initialize the values needed to run the kernel
    int bx=32;
    int by=32;
    dim3 block(bx, by);
    int gx= (width + bx -1)/bx;
    int gy= (height + by -1)/by;
    dim3 grid(gx, gy);

    //Step-1: Conversion from RGB to YCrCb
    cv::Mat h_ycbcr_custom(height, width, CV_8UC3);
    ycbcr* d_ycbcr;
    check(cudaMalloc((void**) &d_ycbcr, ycbcr_bytes));

    cudaEventRecord(start, 0);
    rgb_to_ycbcr<<<grid, block>>>(d_rgb, d_ycbcr, height, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float g_colour_conv;
    cudaEventElapsedTime(&g_colour_conv, start, stop);    
    cout<<"Time taken for colour space conversion: "<< g_colour_conv<<" ms\n";

    check(cudaMemcpy(h_ycbcr_custom.data, d_ycbcr, ycbcr_bytes, cudaMemcpyDeviceToHost));
    //Printing the partial results for a given input
    cout << "\n--- YCbCr Conversion (First 8x8 Block) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            // Assuming h_ycbcr_custom.data directly contains interleaved Y, Cb, Cr bytes
            // If you used h_ycbcr_custom (cv::Mat), then access with .at<cv::Vec3b>(i,j)
            unsigned char y_val = h_ycbcr_custom.data[index * 3];
            unsigned char cb_val = h_ycbcr_custom.data[index * 3 + 1];
            unsigned char cr_val = h_ycbcr_custom.data[index * 3 + 2];
            cout << "(" << (int)y_val << "," << (int)cb_val << "," << (int)cr_val << ") ";
        }
        cout << endl;
    }

    //Step-2: Down sampling- Using average method
    //d_ycbcr contains the image data after converting from RGB->YCrCb format on the device itself. We haven't deleted that yet from the GPU memory, so it can be reused for this operation.
    cv::Mat h_ycbcr_downsampled(height, width, CV_8UC3);
    ycbcr* d_ycbcr_downsampled;
    check(cudaMalloc((void**) &d_ycbcr_downsampled, ycbcr_bytes));

    cudaEventRecord(start, 0);
    downsample<<<grid, block>>>(d_ycbcr, d_ycbcr_downsampled, height, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float g_downsample;
    cudaEventElapsedTime(&g_downsample, start, stop);
    cout<<"Time for downsampling: "<<g_downsample<<" ms \n";
    
    check(cudaMemcpy(h_ycbcr_downsampled.data, d_ycbcr_downsampled, ycbcr_bytes, cudaMemcpyDeviceToHost));
    //saved the downsampled image.
    cv::imwrite("Downsampled.jpg", h_ycbcr_downsampled);

    //Printing partial results
    cout << "\n--- Downsampling (First 8x8 Block) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            unsigned char y_val = h_ycbcr_downsampled.data[index * 3];
            unsigned char cb_val = h_ycbcr_downsampled.data[index * 3 + 1];
            unsigned char cr_val = h_ycbcr_downsampled.data[index * 3 + 2];
            cout << "(" << (int)y_val << "," << (int)cb_val << "," << (int)cr_val << ") ";
        }
        cout << endl;
    }

    //Step-3: DCT
    //Step 3-a: Change the block and grid size suitable for DCT
    //DCT parallel on GPU
    int dct_bx= 8;
    int dct_by=8;
    dim3 dct_block(dct_bx, dct_by);
    int dct_gx= (width + dct_bx-1)/dct_bx;
    int dct_gy= (height + dct_by-1)/dct_by;
    dim3 dct_grid(dct_gx, dct_gy);

    float* d_dct_y;
    float* d_dct_cr;
    float* d_dct_cb;
    int dct_bytes= height * width * sizeof(float);

    check(cudaMalloc((void**) &d_dct_y, dct_bytes));
    check(cudaMalloc((void**) &d_dct_cr, dct_bytes));
    check(cudaMalloc((void**) &d_dct_cb, dct_bytes));
    
    cudaEventRecord(start, 0);
    dct<<<dct_grid, dct_block>>>(d_ycbcr_downsampled, d_dct_y, d_dct_cr, d_dct_cb, height, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float g_dct;
    cudaEventElapsedTime(&g_dct, start, stop);
    cout<<"Time taken for DCT: "<<g_dct<<" ms \n";

    float* h_dct_y = (float*)malloc(dct_bytes);
    float* h_dct_cr = (float*)malloc(dct_bytes);
    float* h_dct_cb = (float*)malloc(dct_bytes);

    check(cudaMemcpy(h_dct_y, d_dct_y, dct_bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_dct_cr, d_dct_cr, dct_bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_dct_cb, d_dct_cb, dct_bytes, cudaMemcpyDeviceToHost));

    //Printing results for the top left block of the image- each for Y, Cr and Cb components
    cout << "\n--- DCT Coefficients (First 8x8 Block - Y Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << fixed << setprecision(2) << h_dct_y[index] << "\t";
        }
        cout << endl;
    }

    cout << "\n--- DCT Coefficients (First 8x8 Block - Cr Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << fixed << setprecision(2) << h_dct_cr[index] << "\t";
        }
        cout << endl;
    }

    cout << "\n--- DCT Coefficients (First 8x8 Block - Cb Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << fixed << setprecision(2) << h_dct_cb[index] << "\t";
        }
        cout << endl;
    }

    //step-4: quantization
    int* q_y, *q_cr, *q_cb;
    size_t qbytes= height * width * sizeof(int);
    cudaMalloc((void**) &q_y,  qbytes);
    cudaMalloc((void**) &q_cr, qbytes);
    cudaMalloc((void**) &q_cb, qbytes);

    int quant_bx= 8;
    int quant_by=8;
    dim3 q_block(quant_bx, quant_by);
    int q_gx= (width+quant_bx-1)/quant_bx;
    int q_gy= (height+quant_by-1)/quant_by;
    dim3 q_grid(q_gx, q_gy);
    
    cudaEventRecord(start, 0);
    quantization<<<q_grid, q_block>>>(q_y, q_cr, q_cb, d_dct_y, d_dct_cr, d_dct_cb, height, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float g_quant;
    cudaEventElapsedTime(&g_quant, start, stop);
    cout<<"Time taken for quantization: "<<g_quant<<" ms \n";

    int* h_q_y = (int*)malloc(qbytes);
    int* h_q_cr = (int*)malloc(qbytes);
    int* h_q_cb = (int*)malloc(qbytes);

    check(cudaMemcpy(h_q_y, q_y, qbytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_q_cr, q_cr, qbytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_q_cb, q_cb, qbytes, cudaMemcpyDeviceToHost));

    //Print results for top left block of the image
    cout << "\n--- Quantized Coefficients (First 8x8 Block - Y Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << h_q_y[index] << "\t";
        }
        cout << endl;
    }

    cout << "\n--- Quantized Coefficients (First 8x8 Block - Cr Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << h_q_cr[index] << "\t";
        }
        cout << endl;
    }

    cout << "\n--- Quantized Coefficients (First 8x8 Block - Cb Component) ---\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int index = i * width + j;
            cout << h_q_cb[index] << "\t";
        }
        cout << endl;
    }
    
    //Step-5: Encoding
    //Step 5-a: Zig-zag scan of matrix in 8*8 blocks.

    int num_blocks_x = (width + 7) / 8;
    int num_blocks_y = (height + 7) / 8;
    size_t total_elements_scanned = num_blocks_x * num_blocks_y * 64; // Each block has 64 elements (8*8)
    size_t zz_bytes = total_elements_scanned * sizeof(int);
    dim3 zz_block(8, 8);
    int zz_gx= (width + 7)/8;
    int zz_gy= (height + 7)/8;
    dim3 zz_grid(dct_gx, dct_gy);
    int* zz_scanned_y, *zz_scanned_cr, *zz_scanned_cb;
    check(cudaMalloc((void**) &zz_scanned_y, zz_bytes));
    check(cudaMalloc((void**) &zz_scanned_cr, zz_bytes));
    check(cudaMalloc((void**) &zz_scanned_cb, zz_bytes));

    cudaEventRecord(start, 0);
    zig_zag_scan<<<zz_grid, zz_block>>>(zz_scanned_y, zz_scanned_cr, zz_scanned_cb, q_y, q_cr, q_cb, height, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float g_zz_scan;
    cudaEventElapsedTime(&g_zz_scan, start, stop);
    cout<<"Time taken for quantization: "<<g_zz_scan<<" ms \n";

    //From this point, there will be no GPU implementation. The following code runs better on the CPU.
    //Step 5-b: RLE
    int* h_zz_y, *h_zz_cr, *h_zz_cb;
    h_zz_y= (int*) malloc(zz_bytes);
    h_zz_cr= (int*) malloc(zz_bytes);
    h_zz_cb= (int*) malloc(zz_bytes);

    //Copy result of zig zag scanning back to the host
    check(cudaMemcpy(h_zz_y, zz_scanned_y, zz_bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_zz_cr, zz_scanned_cr, zz_bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_zz_cb, zz_scanned_cb, zz_bytes, cudaMemcpyDeviceToHost));

    //Print results for top-left block of the image
    cout << "\n--- Zig-Zag Scanned Coefficients (First Block - Y Component) ---\n";
    for (int i = 0; i < 64; ++i) {
        cout << h_zz_y[i] << " ";
        if ((i + 1) % 8 == 0) { // Newline after every 8 elements for better visualization
            cout << endl;
        }
    }
    cout << endl;

    cout << "\n--- Zig-Zag Scanned Coefficients (First Block - Cr Component) ---\n";
    for (int i = 0; i < 64; ++i) {
        cout << h_zz_cr[i] << " ";
        if ((i + 1) % 8 == 0) {
            cout << endl;
        }
    }
    cout << endl;

    cout << "\n--- Zig-Zag Scanned Coefficients (First Block - Cb Component) ---\n";
    for (int i = 0; i < 64; ++i) {
        cout << h_zz_cb[i] << " ";
        if ((i + 1) % 8 == 0) {
            cout << endl;
        }
    }
    cout << endl;

    vector<encoded_symbol> encoded_y_symbols;
    vector<encoded_symbol> encoded_cr_symbols;
    vector<encoded_symbol> encoded_cb_symbols;

    int num_blocks_x1= (width + 7)/8;
    int num_blocks_y1= (height + 7)/8;
    int total_blocks= num_blocks_x1 * num_blocks_y1;

    int prev_dc_y=0, prev_dc_cr=0, prev_dc_cb=0;
    double rle_start= cpu_sec();
    for(int current_block=0; current_block < total_blocks; current_block++){
        int start_index= current_block * 64;
        int diff_dc_y= prev_dc_y - h_zz_y[start_index];
        int dc_y_bits= get_code_bits(diff_dc_y);
        encoded_y_symbols.push_back({0, dc_y_bits, diff_dc_y});
        prev_dc_y = h_zz_y[start_index];

        int diff_dc_cr= prev_dc_cr - h_zz_cr[start_index];
        int dc_cr_bits= get_code_bits(diff_dc_cr);
        encoded_cr_symbols.push_back({0, dc_cr_bits, diff_dc_cr});
        prev_dc_cr = h_zz_cr[start_index];

        int diff_dc_cb= prev_dc_cb - h_zz_cb[start_index];
        int dc_cb_bits= get_code_bits(diff_dc_cb);
        encoded_cb_symbols.push_back({0, dc_cb_bits, diff_dc_cb});
        prev_dc_cb = h_zz_cb[start_index];

        //First: for Y component
        int runlen_y=0;
        for(int i=1; i<64; i++){
            int ac_value_y= h_zz_y[start_index + i];
            if(ac_value_y == 0){
                runlen_y++;
                if(runlen_y == 16){
                    encoded_y_symbols.push_back({15, 0, 0});
                    runlen_y = 0;
                }
            }
            else{
                //append {runlen, non-zero element} to result structure
                int ac_bits_y= get_code_bits(ac_value_y);
                encoded_y_symbols.push_back({runlen_y, ac_bits_y, ac_value_y});
                runlen_y=0;
            }
        }
        //After completion of RLE for a block, need to append the EOB marker
        if (runlen_y > 0 || (h_zz_y[start_index + 63] == 0)) {
            encoded_y_symbols.push_back({0, 0, 0});
        }

        //Second: For Cr component
        int runlen_cr=0;
        for(int i=1; i<64; i++){
            int ac_value_cr= h_zz_cr[start_index + i];
            if(ac_value_cr == 0){
                runlen_cr++;
                if(runlen_cr == 16){
                    encoded_cr_symbols.push_back({15, 0, 0});
                    runlen_cr = 0;
                }
            }
            else{
                //append {runlen, non-zero element} to result structure
                int ac_bits_cr= get_code_bits(ac_value_cr);
                encoded_cr_symbols.push_back({runlen_cr, ac_bits_cr, ac_value_cr});
                runlen_cr=0;
            }
        }
        //After completion of RLE for a block, need to append the EOB marker
        if (runlen_cr > 0 || (h_zz_cr[start_index + 63] == 0)) {
            encoded_cr_symbols.push_back({0, 0, 0}); // EOB symbol (0,0,0)
        }

        //Third: For Cb components
        int runlen_cb=0;
        for(int i=1; i<64; i++){
            int ac_value_cb= h_zz_cb[start_index + i];
            if(ac_value_cb == 0){
                runlen_cb++;
                if(runlen_cb == 16){
                    encoded_cb_symbols.push_back({15, 0, 0});
                    runlen_cb = 0;
                }
            }
            else{
                //append {runlen, non-zero element} to result structure
                int ac_bits_cb= get_code_bits(ac_value_cb);
                encoded_cb_symbols.push_back({runlen_cb, ac_bits_cb, ac_value_cb});
                runlen_cb=0;
            }
        }
        //After completion of RLE for a block, need to append the EOB marker
        if (runlen_cb > 0 || (h_zz_cb[start_index + 63] == 0)) {
            encoded_cb_symbols.push_back({0, 0, 0}); // EOB symbol (0,0,0)
        }
    }
    double rle_end= cpu_sec();
    cout<<"Time for RLE: "<<rle_end-rle_start<<" sec\n";  

    // Add this after: cout<<"Time for RLE: "<<rle_end-rle_start<<" sec\n";

    //Printing partial results
    cout << "\n--- RLE Encoded Symbols (First Block - Y Component) ---\n";
    // Print the DC coefficient of the first block
    if (!encoded_y_symbols.empty()) {
        cout << "DC (Y): {run_len: " << encoded_y_symbols[0].run_len
             << ", size: " << encoded_y_symbols[0].size
             << ", value: " << encoded_y_symbols[0].value << "}\n";
    }
    // Printing a few AC coefficients for the first block
    cout << "AC (Y) - first few:\n";
    int count_ac_y = 0;
    for (size_t i = 1; i < encoded_y_symbols.size() && count_ac_y < 10; ++i) { // Print up to 10 AC symbols
        // Stop if we hit EOB for the first block (value=0, size=0, run_len=0)
        if (encoded_y_symbols[i].run_len == 0 && encoded_y_symbols[i].size == 0 && encoded_y_symbols[i].value == 0) {
            cout << "  EOB\n";
            break;
        }
        cout << "  {run_len: " << encoded_y_symbols[i].run_len
             << ", size: " << encoded_y_symbols[i].size
             << ", value: " << encoded_y_symbols[i].value << "}\n";
        count_ac_y++;
    }

    cout << "\n--- RLE Encoded Symbols (First Block - Cr Component) ---\n";
    // Print the DC coefficient of the first block
    if (encoded_cr_symbols.size() >= (size_t)1) { // Ensure there's a first DC symbol
        cout << "DC (Cr): {run_len: " << encoded_cr_symbols[0].run_len
             << ", size: " << encoded_cr_symbols[0].size
             << ", value: " << encoded_cr_symbols[0].value << "}\n";
    }
    // Print a few AC coefficients for the first block
    cout << "AC (Cr) - first few:\n";
    int count_ac_cr = 0;
    for (size_t i = 1; i < encoded_cr_symbols.size() && count_ac_cr < 10; ++i) {
        if (encoded_cr_symbols[i].run_len == 0 && encoded_cr_symbols[i].size == 0 && encoded_cr_symbols[i].value == 0) {
            cout << "  EOB\n";
            break;
        }
        cout << "  {run_len: " << encoded_cr_symbols[i].run_len
             << ", size: " << encoded_cr_symbols[i].size
             << ", value: " << encoded_cr_symbols[i].value << "}\n";
        count_ac_cr++;
    }

    cout << "\n--- RLE Encoded Symbols (First Block - Cb Component) ---\n";
    // Print the DC coefficient of the first block
    if (encoded_cb_symbols.size() >= (size_t)1) { // Ensure there's a first DC symbol
        cout << "DC (Cb): {run_len: " << encoded_cb_symbols[0].run_len
             << ", size: " << encoded_cb_symbols[0].size
             << ", value: " << encoded_cb_symbols[0].value << "}\n";
    }
    // Print a few AC coefficients for the first block
    cout << "AC (Cb) - first few:\n";
    int count_ac_cb = 0;
    for (size_t i = 1; i < encoded_cb_symbols.size() && count_ac_cb < 10; ++i) {
        if (encoded_cb_symbols[i].run_len == 0 && encoded_cb_symbols[i].size == 0 && encoded_cb_symbols[i].value == 0) {
            cout << "  EOB\n";
            break;
        }
        cout << "  {run_len: " << encoded_cb_symbols[i].run_len
             << ", size: " << encoded_cb_symbols[i].size
             << ", value: " << encoded_cb_symbols[i].value << "}\n";
        count_ac_cb++;
    }

    //Step 5-c: Huffman coding

    // DC Luminance (Y) Huffman Table
    // BITS array: How many Huffman codes there are for each possible bit length.
    const unsigned char DC_L_BITS[] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    // HUFFVAL array: lists the symbols that are being encoded.
    const unsigned char DC_L_HUFFVAL[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    // AC Luminance (Y) Huffman Table (Predefined)
    const unsigned char AC_L_BITS[] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 4, 4, 0, 0, 1, 125, 0};
    const unsigned char AC_L_HUFFVAL[] = {
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
        0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    };

    // DC Chrominance (Cr/Cb) Huffman Table
    const unsigned char DC_C_BITS[] = {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    const unsigned char DC_C_HUFFVAL[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    // AC Chrominance (Cr/Cb) Huffman Table (Predefined)
    const unsigned char AC_C_BITS[] = {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119};
    const unsigned char AC_C_HUFFVAL[] = {
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
        0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
        0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
        0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
        0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    };
// Building Huffman Tables
    std::map<unsigned char, std::pair<unsigned short, unsigned char>> dc_l_huff_table = build_huffman_table(DC_L_BITS, DC_L_HUFFVAL);
    std::map<unsigned char, std::pair<unsigned short, unsigned char>> ac_l_huff_table = build_huffman_table(AC_L_BITS, AC_L_HUFFVAL);
    std::map<unsigned char, std::pair<unsigned short, unsigned char>> dc_c_huff_table = build_huffman_table(DC_C_BITS, DC_C_HUFFVAL);
    std::map<unsigned char, std::pair<unsigned short, unsigned char>> ac_c_huff_table = build_huffman_table(AC_C_BITS, AC_C_HUFFVAL);

    BitStreamWriter writer;
    double huff_start = cpu_sec();

    // Encode Y component
    for (const auto& symbol : encoded_y_symbols) {
        unsigned char huffman_symbol;
        std::map<unsigned char, std::pair<unsigned short, unsigned char>>* current_huff_table;

        if (symbol.run_len == 0) { // This covers both DC and EOB/ZRL if size is 0 and value is 0
            if (symbol.size == 0 && symbol.value == 0) { // EOB (End of Block)
                huffman_symbol = 0x00;
                current_huff_table = &ac_l_huff_table; // EOB is an AC code
            } else { // DC coefficient
                huffman_symbol = static_cast<unsigned char>(symbol.size);
                current_huff_table = &dc_l_huff_table;
            }
        } else if (symbol.run_len == 15 && symbol.size == 0 && symbol.value == 0) { // ZRL (Zero Run Length)
            huffman_symbol = 0xF0;
            current_huff_table = &ac_l_huff_table; // ZRL is an AC code
        } else { // AC coefficient
            huffman_symbol = (static_cast<unsigned char>(symbol.run_len) << 4) | static_cast<unsigned char>(symbol.size);
            current_huff_table = &ac_l_huff_table;
        }

        auto it = current_huff_table->find(huffman_symbol);
        if (it != current_huff_table->end()) {
            writer.write_bits(it->second.first, it->second.second); // Huffman code
            // Only write additional bits for non-EOB/ZRL symbols
            if (huffman_symbol != 0x00 && huffman_symbol != 0xF0) {
                writer.write_bits(get_value_bits(symbol.value, symbol.size), symbol.size); // Additional bits
            }
        } else {
            std::cerr << "Error: Huffman symbol " << (int)huffman_symbol << " not found for Y component!" << std::endl;
        }
    }

    // Encode Cr component
    for (const auto& symbol : encoded_cr_symbols) {
        unsigned char huffman_symbol;
        std::map<unsigned char, std::pair<unsigned short, unsigned char>>* current_huff_table;

        if (symbol.run_len == 0) {
            if (symbol.size == 0 && symbol.value == 0) { // EOB
                huffman_symbol = 0x00;
                current_huff_table = &ac_c_huff_table;
            } else { // DC coefficient
                huffman_symbol = static_cast<unsigned char>(symbol.size);
                current_huff_table = &dc_c_huff_table;
            }
        } else if (symbol.run_len == 15 && symbol.size == 0 && symbol.value == 0) { // ZRL
            huffman_symbol = 0xF0;
            current_huff_table = &ac_c_huff_table;
        } else { // AC coefficient
            huffman_symbol = (static_cast<unsigned char>(symbol.run_len) << 4) | static_cast<unsigned char>(symbol.size);
            current_huff_table = &ac_c_huff_table;
        }

        auto it = current_huff_table->find(huffman_symbol);
        if (it != current_huff_table->end()) {
            writer.write_bits(it->second.first, it->second.second);
            if (huffman_symbol != 0x00 && huffman_symbol != 0xF0) {
                writer.write_bits(get_value_bits(symbol.value, symbol.size), symbol.size);
            }
        } else {
            std::cerr << "Error: Huffman symbol " << (int)huffman_symbol << " not found for Cr component!" << std::endl;
        }
    }

    // Encode Cb component
    for (const auto& symbol : encoded_cb_symbols) {
        unsigned char huffman_symbol;
        std::map<unsigned char, std::pair<unsigned short, unsigned char>>* current_huff_table;

        if (symbol.run_len == 0) {
            if (symbol.size == 0 && symbol.value == 0) { // EOB
                huffman_symbol = 0x00;
                current_huff_table = &ac_c_huff_table;
            } else { // DC coefficient
                huffman_symbol = static_cast<unsigned char>(symbol.size);
                current_huff_table = &dc_c_huff_table;
            }
        } else if (symbol.run_len == 15 && symbol.size == 0 && symbol.value == 0) { // ZRL
            huffman_symbol = 0xF0;
            current_huff_table = &ac_c_huff_table;
        } else { // AC coefficient
            huffman_symbol = (static_cast<unsigned char>(symbol.run_len) << 4) | static_cast<unsigned char>(symbol.size);
            current_huff_table = &ac_c_huff_table;
        }

        auto it = current_huff_table->find(huffman_symbol);
        if (it != current_huff_table->end()) {
            writer.write_bits(it->second.first, it->second.second);
            if (huffman_symbol != 0x00 && huffman_symbol != 0xF0) {
                writer.write_bits(get_value_bits(symbol.value, symbol.size), symbol.size);
            }
        } else {
            std::cerr << "Error: Huffman symbol " << (int)huffman_symbol << " not found for Cb component!" << std::endl;
        }
    }

    writer.flush_bits();
    double huff_end = cpu_sec();
    double huff_time = huff_end - huff_start;

    cout << "Huffman encoding time: " << huff_time * 1000.0 << " ms" << endl;
    cout << "Compressed data size: " << writer.data.size() << " bytes" << endl;

    cudaFree(d_rgb); cudaFree(d_ycbcr); cudaFree(d_ycbcr_downsampled);
    cudaFree(d_dct_y); cudaFree(d_dct_cr); cudaFree(d_dct_cb);
    cudaFree(q_cb); cudaFree(q_cr); cudaFree(q_y);
    cudaFree(zz_scanned_y); cudaFree(zz_scanned_cr); cudaFree(zz_scanned_cb);
    free(h_zz_y); free(h_zz_cr); free(h_zz_cb);
    free(h_dct_y); free(h_dct_cr); free(h_dct_cb);
    free(h_q_y); free(h_q_cr); free(h_q_cb);
    return 0;
}
