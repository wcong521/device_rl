extern "C" {
    #include <stdio.h>
    __global__ void test(float* arr, int n)                                                       
    {                                                            
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        printf("%d", n);
        if (tid < n) {
            arr[tid] += 1;
        }                              
    }
}