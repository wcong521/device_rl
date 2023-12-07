extern "C" {
    #include <stdio.h>
    __global__ void step(float* observations, float* actions)                                                       
    {            
        int env_idx = blockIdx.x;
        int agent_idx = threadIdx.x;

        observations[env_idx * blockDim.x * 2 + agent_idx * 2 + 0] += actions[env_idx * blockDim.x * 2 + agent_idx * 2 + 0];
        observations[env_idx * blockDim.x * 2 + agent_idx * 2 + 1] += actions[env_idx * blockDim.x * 2 + agent_idx * 2 + 1];

        // int tid = blockDim.x * blockIdx.x + threadIdx.x;
        // printf("%d\n", n);
        // if (tid < n) {
        //     arr[tid] += 1;
        // }                              
    }
}