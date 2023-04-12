#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        #define BLOCK_SIZE 512
        // TODO: 
        /**
         * Koggle Stone **inclusive** parallel scan using double buffering.
         * 
         */ 
        __global__ void scan_ks_kernel(int n, int* dev_odata, const int* dev_idata) {
            // double buffering
            __shared__ int T0[BLOCK_SIZE];
            __shared__ int T1[BLOCK_SIZE];

            int bi = blockIdx.x;
            int ti = threadIdx.x;
            int index = bi * blockDim.x + ti;

            int *src = T0;
            int *dest = T1;

            if (index < n) {
                T0[ti] = dev_idata[index];
                T1[ti] = T0[ti];
            }

            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
                __syncthreads();
                if (ti >= stride)
                    dest[ti] = src[ti] + src[ti - stride];
                else
                    dest[ti] = src[ti];
                int* tmp = src;
                src = dest;
                dest = tmp;
            }
            if (index < n) {
                dev_odata[index] = src[ti];
            }
        }

        /**
         * Addition on partial sum per block
         */
        __global__ void add_kernel(int n, int* dev_odata, const int* dev_idata) {
            int bi = blockIdx.x;
            int ti = threadIdx.x;
            int index = (bi + 1) * blockDim.x + ti;
            if (index < n) {
                dev_odata[index] += dev_idata[bi];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            unsigned int grid_size = (n - 1 + BLOCK_SIZE) / BLOCK_SIZE;
            dim3 gridDim = { grid_size, 1, 1 };
            dim3 blockDim = { BLOCK_SIZE, 1, 1 };
            timer().startGpuTimer();
            // TODO
            // step 1 scan into partial prefix-sum for each block
            scan_ks_kernel <<< gridDim, blockDim >>> (n, dev_odata, dev_idata);
            
            // step 2 perform scan on the last element in each block
            int* dev_blockSumsInput;
            int* host_blockSumsInput;
            int* dev_blockSumsOutput;
            // memory allocation
            host_blockSumsInput = (int*)malloc(grid_size * sizeof(int));
            cudaMalloc((void**)&dev_blockSumsInput, grid_size * sizeof(int));
            cudaMalloc((void**)&dev_blockSumsOutput, grid_size * sizeof(int));
            // memory copy
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < grid_size; i++)
                host_blockSumsInput[i] = odata[(i + 1) * BLOCK_SIZE - 1];
            cudaMemcpy(dev_blockSumsInput, host_blockSumsInput, grid_size * sizeof(int), \
                cudaMemcpyHostToDevice);
            scan_ks_kernel<<< 1, grid_size >>>(grid_size, dev_blockSumsOutput, \
                dev_blockSumsInput);
            // step3 add scanned block prefix-sum i into all values in block i+1
            add_kernel <<< gridDim, blockDim >>> (n, dev_odata, dev_blockSumsOutput);
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            // since the scan kernel is inclusive, we have to set the first element to 0
            cudaMemcpy(odata+1, dev_odata, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            checkCUDAError("navie scan");
        }
    }
}
