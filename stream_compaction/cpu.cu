#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            // timer().startCpuTimer();
            // TODO
            if (n < 1) return;
            odata[0] = 0;
            for (int i = 1; i < n; i++)
                odata[i] = odata[i - 1] + idata[i - 1];
            // timer().endCpuTimer();
        }

        /**
         * CPU scatter
         * @returns the number of elements in the result array
         */
        int scatter(int n, int* odata, const int* idata, const int* tdata, const int* sdata) {
            // timer().startCpuTimer();
            // TODO
            if (n < 1) return -1;
            int num_elements = 0;
            for (int i = 0; i < n; i++) {
                if (tdata[i] == 1) {
                    num_elements++;
                    odata[sdata[i]] = idata[i];
                }
            }
            // timer().endCpuTimer();
            return num_elements;
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n < 1) return -1;
            int num_nozeros = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[num_nozeros++] = idata[i];
                    // num_nozeros += 1;
                }
            }
            timer().endCpuTimer();
            return num_nozeros;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n < 1) return -1;

            // map input data into temporary array
            int* tdata = new int[n]; // temporary array for 0s and 1s
            for (int i = 0; i < n; i++) {
                tdata[i] = 0;
            }
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0)
                    tdata[i] = 0;
                else
                    tdata[i] = 1;
            }

            // perform scan on temporary array
            int* sdata = new int[n];
            for (int i = 0; i < n; i++) {
                sdata[i] = 0;
            }
            scan(n, sdata, tdata);
            
            // scatter using the result of scan as write index
            int num_elements = scatter(n, odata, idata, tdata, sdata);
            timer().endCpuTimer();
            return num_elements;
        }
    }
}
