extern "C" {
    #include <stdio.h>
    __global__ void step(float* observations, float* actions)                                                       
    {            
        int env_idx = blockIdx.x;
        int agent_idx = threadIdx.x;

        observations[env_idx * blockDim.x + agent_idx + 0] += actions[env_idx * blockDim.x + agent_idx + 0];
        observations[env_idx * blockDim.x + agent_idx + 1] += actions[env_idx * blockDim.x + agent_idx + 1];

        // int tid = blockDim.x * blockIdx.x + threadIdx.x;
        // printf("%d\n", n);
        // if (tid < n) {
        //     arr[tid] += 1;
        // }                              
    }
}