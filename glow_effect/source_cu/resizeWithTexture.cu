// resizeWithTexture.cu
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

__global__ void resizeKernel(cudaTextureObject_t texObj, float* output,
                             int outWidth, int outHeight,
                             int inWidth, int inHeight, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < outWidth && y < outHeight)
    {
        // Map output pixel (x,y) to input coordinate (u,v)
        float u = (x + 0.5f) * inWidth / outWidth - 0.5f;
        float v = (y + 0.5f) * inHeight / outHeight - 0.5f;
        float4 pixel = tex2D<float4>(texObj, u, v);
        int index = (y * outWidth + x) * channels;
        output[index + 0] = pixel.x;
        output[index + 1] = pixel.y;
        output[index + 2] = pixel.z;
        output[index + 3] = pixel.w;
    }
}

extern "C" void resizeWithTexture(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                    int outWidth, int outHeight, cudaStream_t stream)
{
    // Get input dimensions.
    int inWidth  = input.cols;
    int inHeight = input.rows;
    int channels = input.channels();  // must be 4 for float4.

    // Create a CUDA array and copy the input into it.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, inWidth, inHeight);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocArray error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy2DToArray(cuArray, 0, 0, input.ptr(), input.step,
                              inWidth * sizeof(float4), inHeight, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy2DToArray error: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return;
    }

    // Set up the resource and texture descriptors.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use unnormalized coordinates

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaCreateTextureObject error: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return;
    }

    // Allocate output GPU memory. Make sure the output GpuMat is CV_32FC4.
    output.create(outHeight, outWidth, input.type());
    float* d_output = reinterpret_cast<float*>(output.ptr());

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);
    resizeKernel<<<grid, block, 0, stream>>>(texObj, d_output, outWidth, outHeight,
                                              inWidth, inHeight, channels);
    cudaStreamSynchronize(stream);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}
