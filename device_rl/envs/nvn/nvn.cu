#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

extern "C" {

    __global__ void reset(
        float* observations, 
        float* agents, 
        float* opponents, 
        float* ball, 
        int* goal_scored
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int env_idx = blockIdx.x;
        int agent_idx = threadIdx.x;

        curandState_t state;
        curand_init(7, tid, 0, &state);

        // float random_float = 9000.0f * (curand_uniform(&state) - 0.5f);

        int agents_inx = env_idx * blockDim.x * 4 + agent_idx * 4;
        agents[agents_inx + 0] = 9000.0f * (curand_uniform(&state) - 0.5f);
        agents[agents_inx + 1] = 9000.0f * (curand_uniform(&state) - 0.5f);
        agents[agents_inx + 2] = 2.0f * M_PI * (curand_uniform(&state) - 0.5f);
        agents[agents_inx + 3] = 0.f;


        // int tid = blockDim.x * blockIdx.x + threadIdx.x;
        // printf("%d\n", n);
        // if (tid < n) {
        //     arr[tid] += 1;
        // }                              
    }
}