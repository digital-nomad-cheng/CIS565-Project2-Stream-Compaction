#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::device_vector<int> device_idata(idata, idata+n);
            thrust::device_vector<int> device_odata;
            device_odata.resize(n);
            timer().startGpuTimer();
            thrust::exclusive_scan(device_idata.begin(), device_idata.end(), device_odata.begin());
            timer().endGpuTimer();
            for (int i = 0; i < n; i++)
                odata[i] = device_odata[i];
            
        }
    }
}
