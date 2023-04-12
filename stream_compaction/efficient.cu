#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Brent - kung parallel scan : O(n)
         */
        __global__ void scan_bk_kernel(int n, int* dev_odata, const int* dev_idata) {
            // double buffering
            __shared__ float T[2 * BLOCK_SIZE];
            int bi = blockIdx.x;
            int ti = threadIdx.x;
            int start_idx = 2 * bi * blockDim.x;
            // each thread need two load two elements into shared memory
            if (ti + start_idx < n)
                T[ti] = dev_idata[start_idx + ti];
            if (ti + start_idx + blockDim.x < n)
                T[ti + blockDim.x] = dev_idata[start_idx + blockDim.x + ti];

            // reduction step 
            int stride = 1;
            while (stride < 2 * BLOCK_SIZE) {
                __syncthreads();
                int idx = (ti + 1) * stride * 2 - 1;
                if (idx < 2 * BLOCK_SIZE && (idx - stride) >= 0)
                    T[idx] += T[idx - stride];
                stride *= 2;
            }

            // post scan step
            stride = BLOCK_SIZE / 2;
            while (stride > 0) {
                __syncthreads();
                int idx = (ti + 1) * stride * 2 - 1;
                if ((idx + stride) < 2 * BLOCK_SIZE)
                    T[idx + stride] += T[idx];
                stride /= 2;
            }

            if (ti + start_idx < n)
                dev_odata[ti + start_idx] = T[ti];
            if (ti + start_idx + blockDim.x < n)
                dev_odata[ti + start_idx + blockDim.x] = T[ti + blockDim.x];
        }

        /**
         * Addition on partial sum per block
         */
        __global__ void add_kernel(int n, int* dev_odata, const int* dev_idata) {
            int bi = blockIdx.x;
            int ti = threadIdx.x;
            int start_idx = 2 * (bi + 1) * blockDim.x;
            int index = start_idx + ti;
            if (index < n) {
                dev_odata[index] += dev_idata[bi];
            }
            if (index + blockDim.x < n) {
                dev_odata[index + blockDim.x] += dev_idata[bi];
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
            unsigned int grid_size = (n - 1) / ( 2 * BLOCK_SIZE) + 1;
            dim3 gridDim = { grid_size, 1, 1 };
            dim3 blockDim = { BLOCK_SIZE, 1, 1 };
            timer().startGpuTimer();
            // TODO
            // step 1 scan into partial prefix-sum for each block
            scan_bk_kernel <<< gridDim, blockDim >>> (n, dev_odata, dev_idata);

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
            for (unsigned int i = 0; i < grid_size; i++)
                host_blockSumsInput[i] = odata[(i + 1) * 2 * BLOCK_SIZE - 1];
            cudaMemcpy(dev_blockSumsInput, host_blockSumsInput, grid_size * sizeof(int), \
                cudaMemcpyHostToDevice);
            scan_bk_kernel <<< 1, (grid_size-1)/2 + 1 >>> (grid_size, dev_blockSumsOutput, \
                dev_blockSumsInput);
            // step3 add scanned block prefix-sum i into all values in block i+19
            add_kernel <<< gridDim, blockDim >>> (n, dev_odata, dev_blockSumsOutput);
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            // since the scan kernel is inclusive, we have to set the first element to 0
            cudaMemcpy(odata + 1, dev_odata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            checkCUDAError("efficient scan");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // timer().startGpuTimer();
            // TODO
            // timer().endGpuTimer();
            return -1;
        }
    }
}
