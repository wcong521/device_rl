#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

extern "C" {

    __device__ float random_uniform(
        curandState_t *rng,
        float low,
        float high
    ) {
        return low + (high - low) * curand_uniform(rng);
    }

    __device__ void get_rel_obs(

    ) {
        return;
    }

    __device__ float* get_obs(

    ) {
        return;
    }

    __global__ void reset(
        float* obs, 
        float* state, 
        int num_agents,
        int num_opponents
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int env_idx = blockIdx.x;
        // int agent_idx = threadIdx.x;

        curandState_t rng;
        curand_init(7, tid, 0, &rng);

        int i = env_idx * blockDim.x * 4 + threadIdx.x * 4;

        // agents
        if (threadIdx.x < num_agents) {

            state[i + 0] = random_uniform(&rng, -4500, 4500);
            state[i + 1] = random_uniform(&rng, -3000, 3000);
            state[i + 2] = random_uniform(&rng, -M_PI, M_PI);
            state[i + 3] = 0.f;

        // opponents
        } else if (threadIdx.x < num_agents + num_opponents) {

            state[i + 0] = random_uniform(&rng, -4500, 4500);
            state[i + 1] = random_uniform(&rng, -3000, 3000);
            state[i + 2] = random_uniform(&rng, -M_PI, M_PI);
            state[i + 3] = 0.f;

        // ball
        } else {

            state[i + 0] = random_uniform(&rng, -4500, 4500);
            state[i + 1] = random_uniform(&rng, -3000, 3000);
            state[i + 2] = 0.f;
            state[i + 3] = 0.f;

        }   
                                
    }

    __global__ void sample(
        float* actions
    ) {
        return;
    }

    __global__ void step(
        float* state, 
        float* act,
        int num_agents,
        int num_opponents,
        float* obs,
        float* rew,
        bool* term,
        bool* trunc,
        float* info
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int env_idx = blockIdx.x;
        // int agent_idx = threadIdx.x;
        int i = env_idx * blockDim.x * 4 + threadIdx.x * 4;

        curandState_t rng;
        // TODO: figure out how to make this properly random
        curand_init(__float2int_rd(state[i]), tid, 0, &rng);

        int step_size = 50;

        // agents
        if (threadIdx.x < num_agents) {

            state[i + 0] += random_uniform(&rng, -step_size, step_size);
            state[i + 1] += random_uniform(&rng, -step_size, step_size);

        // opponents
        } else if (threadIdx.x < num_agents + num_opponents) {

            state[i + 0] += random_uniform(&rng, -step_size, step_size);
            state[i + 1] += random_uniform(&rng, -step_size, step_size);

        // ball
        } else {

            state[i + 0] += random_uniform(&rng, -step_size, step_size);
            state[i + 1] += random_uniform(&rng, -step_size, step_size);

        }                           
    }

    __global__ void test(
        
    ) {
        return;
    }



}