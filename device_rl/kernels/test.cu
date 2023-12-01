extern "C" {
    #include <stdio.h>
    __global__ void test(int n)                                                       
    {                                                            
        // int tid = blockDim.x * blockIdx.x + threadIdx.x;
        printf("%d\n", n);
        // if (tid < n) {
        //     arr[tid] += 1;
        // }                              
    }
}