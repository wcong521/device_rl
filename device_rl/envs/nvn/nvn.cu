#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

__constant__ float X_DISP = 1.5;
__constant__ float Y_DISP = 1.5;
__constant__ float ANGLE_DISP = 1.2;

__constant__ float MAX_X_VEL = 2.;
__constant__ float MAX_Y_VEL = 2.;
__constant__ float MAX_ANGLE_VEL = 2.;

__constant__ float DISPLACEMENT_COEF = 0.06;
__constant__ float ANGLE_DISPLACEMENT = 0.05;

__constant__ float BOUNCE_MULTIPLIER = 10;
__constant__ float ROBOT_RADIUS = 20;

__constant__ float BALL_RADIUS = 10;
__constant__ float BALL_ACCELERATION = -0.8;
__constant__ float BALL_VELOCITY_COEF = 1;
__constant__ float BALL_BOUNCE_MULTIPLIER = 1;


extern "C" {

    __device__ float random_uniform(
        curandState_t* rng,
        float low,
        float high
    ) {
        return low + (high - low) * curand_uniform(rng);
    }

    __device__ float random_normal(
        curandState_t* rng,
        float mean,
        float std
    ) {
        return curand_normal(rng) * std + mean;
    }

    __device__ float clip(
        float value,
        float low,
        float high
    ) {
        return fmaxf(low, fminf(value, high));
    }


    __device__ void get_rel_obs(

    ) {
        return;
    }


    __device__ void check_collisions(
        float* state,
        int state_size
    ) {
        int state_idx = threadIdx.x * state_size;

        // -1 for ball
        int num_entities = blockDim.x - 1;
        for (int i = 0; i < num_entities; i++) {

            // skip self
            if (i == threadIdx.x) continue;

            int other_state_idx = i * state_size;

            // compute distance between the two entities
            float dx = state[other_state_idx + 0] - state[state_idx + 0];
            float dy = state[other_state_idx + 1] - state[state_idx + 1];
            float distance = sqrtf(dx * dx + dy * dy);

            // skip if no collision
            if (distance >= (2 * ROBOT_RADIUS) * 7) continue;

            // angle between entities
            float angle = atan2f(dy, dx);

            // bounce velocity
            float x_vel = cosf(angle) * BOUNCE_MULTIPLIER;
            float y_vel = sinf(angle) * BOUNCE_MULTIPLIER;

            // use atomic operations here to prevent race conditions

            // update self
            atomicExch(&state[state_idx + 3], -x_vel);
            atomicExch(&state[state_idx + 4], -y_vel);
            atomicAdd(&state[state_idx + 0], -x_vel);
            atomicAdd(&state[state_idx + 1], -y_vel);

            // update other
            atomicExch(&state[other_state_idx + 3], x_vel);
            atomicExch(&state[other_state_idx + 4], y_vel);
            atomicAdd(&state[other_state_idx + 0], x_vel);
            atomicAdd(&state[other_state_idx + 1], y_vel);
        }
    }

    __device__ void update_ball(
        float* state,
        int state_size,
        int seed
    ) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // update ball velocity
        int ball_idx = threadIdx.x * state_size;
        state[ball_idx + 2] += BALL_ACCELERATION;
        state[ball_idx + 2] = clip(state[ball_idx + 2], 0, 100);

        // update ball position
        state[ball_idx + 0] += state[ball_idx + 2] * cosf(state[ball_idx + 3]);
        state[ball_idx + 1] += state[ball_idx + 2] * sinf(state[ball_idx + 3]);

        // -1 for ball
        int num_entities = blockDim.x - 1;
        for (int i = 0; i < num_entities; i++) {

            int entity_idx = i * state_size;

            // compute distance between ball and entity
            float dx = state[ball_idx + 0] - state[entity_idx + 0];
            float dy = state[ball_idx + 1] - state[entity_idx + 1];
            float distance = sqrtf(dx * dx + dy * dy);

            // skip if no collision
            if (distance >= (ROBOT_RADIUS + BALL_RADIUS) * 6) continue;

            curandState_t rng;
            curand_init(__float2int_rd(seed + tid), tid, 0, &rng);

            state[ball_idx + 2] = BALL_VELOCITY_COEF * 10;
            state[ball_idx + 3] = atan2f(dy, dx);
            state[ball_idx + 3] += random_normal(&rng, -1, 1) * M_PI / 8;

        }

        bool bounce = true;
        int ball_x_sign = state[ball_idx + 0] < 0 ? -1 : 1;
        int ball_y_sign = state[ball_idx + 1] < 0 ? -1 : 1;
        
        if (abs(state[ball_idx + 1]) > 3000) {
            if (bounce) {

                state[ball_idx + 1] = ball_y_sign * 3000;
                state[ball_idx + 2] *= BALL_BOUNCE_MULTIPLIER;
                state[ball_idx + 3] *= -1;

            } else {

                // TODO 

            }
        }

        if (abs(state[ball_idx + 0]) > 4500 && abs(state[ball_idx + 1]) > 1100) {
            if (bounce) {

                state[ball_idx + 0] = ball_x_sign * 4500;
                state[ball_idx + 2] *= BALL_BOUNCE_MULTIPLIER;
                state[ball_idx + 3] = M_PI - state[ball_idx + 3];

            } else {

                // TODO 

            }
        }


    }

    __device__ float* get_obs(

    ) {
        return;
    }

    __device__ void move_agent(
        float* state,
        float* action
    ) {

        if (action[3] > 0.8) {
            // kick
            return;
        } 
        
        // update velocities
        state[3] += clip(action[0] * X_DISP, -0.3 * MAX_X_VEL, MAX_X_VEL);
        state[4] += clip(action[1] * Y_DISP, -0.5 * MAX_Y_VEL, 0.5 * MAX_Y_VEL);
        state[5] += clip(action[2] * ANGLE_DISP, -0.5 * MAX_ANGLE_VEL, 0.5 * MAX_ANGLE_VEL);

        // the target location of the action
        float policy_goal_x = state[0] + (cosf(state[2]) * state[3] + (cosf(state[2]) + M_PI / 2) * state[4]) * 100;
        float policy_goal_y = state[1] + (sinf(state[2]) * state[3] + (sinf(state[2]) + M_PI / 2) * state[4]) * 100;

        // update pose
        // weighted sum of current pose and target pose
        // the idea is that we move towards the target pose
        state[0] = state[0] * (1 - DISPLACEMENT_COEF) + policy_goal_x * DISPLACEMENT_COEF;
        state[1] = state[1] * (1 - DISPLACEMENT_COEF) + policy_goal_y * DISPLACEMENT_COEF;
        state[2] = state[2] + state[5] * ANGLE_DISPLACEMENT;

        // make sure agent is on the field
        state[0] = clip(state[0], -5200, 5200);
        state[1] = clip(state[1], -3700, 3700);
    }

    __device__ void move_opponent(
        float* state,
        float* action
    ) {
        move_agent(state, action);
    }

    __global__ void reset(
        float* state, 
        int state_size,
        int num_agents,
        int num_opponents,
        float* obs,
        int seed
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int state_idx = blockIdx.x * blockDim.x * state_size + threadIdx.x * state_size;

        curandState_t rng;
        curand_init(seed + tid, tid, 0, &rng);

        // agents
        if (threadIdx.x < num_agents) {

            state[state_idx + 0] = random_uniform(&rng, -4500, 0);
            state[state_idx + 1] = random_uniform(&rng, -3000, 3000);
            state[state_idx + 2] = random_uniform(&rng, -M_PI, M_PI);
            state[state_idx + 3] = 0.f;

        // opponents
        } else if (threadIdx.x < num_agents + num_opponents) {

            state[state_idx + 0] = random_uniform(&rng, -4500, 0);
            state[state_idx + 1] = random_uniform(&rng, -3000, 3000);
            state[state_idx + 2] = random_uniform(&rng, -M_PI, M_PI);
            state[state_idx + 3] = 0.f;

        // ball
        } else {

            state[state_idx + 0] = random_uniform(&rng, 0, 4500);
            state[state_idx + 1] = random_uniform(&rng, -3000, 3000);
            state[state_idx + 2] = 0.f;
            state[state_idx + 3] = 0.f;

        } 

        // // agents
        // if (threadIdx.x < num_agents) {

        //     state[state_idx + 0] = blockIdx.x;
        //     state[state_idx + 1] = blockIdx.x;
        //     state[state_idx + 2] = blockIdx.x;
        //     state[state_idx + 3] = 0.f;

        // // opponents
        // } else if (threadIdx.x < num_agents + num_opponents) {

        //     state[state_idx + 0] = blockIdx.x;
        //     state[state_idx + 1] = blockIdx.x;
        //     state[state_idx + 2] = blockIdx.x;
        //     state[state_idx + 3] = 0.f;

        // // ball
        // } else {

        //     state[state_idx + 0] = blockIdx.x;
        //     state[state_idx + 1] = blockIdx.x;
        //     state[state_idx + 2] = 0.f;
        //     state[state_idx + 3] = 0.f;

        // }     
                                
    }

    __global__ void sample(
        float* action,
        int action_size,
        int seed
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int action_idx = blockIdx.x * (blockDim.x - 1) * action_size + threadIdx.x * action_size;
        int step_size = 1;

        curandState_t rng;
        curand_init(__float2int_rd(seed + tid), tid, 0, &rng);

        // action[action_idx + 0] = random_uniform(&rng, -step_size, step_size);
        // action[action_idx + 1] = random_uniform(&rng, -step_size, step_size);
        // action[action_idx + 2] = random_uniform(&rng, -M_PI, M_PI);

        action[action_idx + 0] = 0;
        action[action_idx + 1] = 0;
        action[action_idx + 2] = 0;    

        // action[action_idx + 0] = 0;
        // action[action_idx + 1] = 0;
        // action[action_idx + 2] = 0;
        
    }

    __global__ void step(
        float* state, 
        int state_size,
        float* action,
        int action_size,
        int num_agents,
        int num_opponents,
        int seed,
        float* obs,
        float* rew,
        bool* term,
        bool* trunc,
        float* info
    ) {            

        extern __shared__ float sh_mem[];
        float* sh_state = &sh_mem[0];
        float* sh_action = &sh_mem[gridDim.x * blockDim.x * state_size];

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        int sh_state_idx = threadIdx.x * state_size;
        int sh_action_idx = threadIdx.x * action_size;

        int state_idx = blockIdx.x * blockDim.x * state_size + sh_state_idx;
        int action_idx = blockIdx.x * (blockDim.x - 1) * action_size + sh_action_idx;

        // load shared memory
        for (int i = 0; i < state_size; i++) {
            sh_state[sh_state_idx + i] = state[state_idx + i];
        }

        for (int i = 0; i < action_size; i++) {
            sh_action[sh_action_idx + i] = action[action_idx + i];
        }

        // ensure that all threads have finished loading shared memory before continuing
        __syncthreads();

        if (threadIdx.x < num_agents) {
            move_agent(&sh_state[sh_state_idx], &sh_action[sh_action_idx]);
        } else if (threadIdx.x < num_agents + num_opponents) {
            move_opponent(&sh_state[sh_state_idx], &sh_action[sh_action_idx]);
        } 

        __syncthreads();

        if (threadIdx.x < num_agents + num_opponents) {
            check_collisions(sh_state, state_size);
        } else {
            // last thread updates ball
            update_ball(sh_state, state_size, seed);
        }

        // ensure that all threads have finished moditfying shared memory before continuing
        __syncthreads();

        for (int i = 0; i < state_size; i++)
            state[state_idx + i] = sh_state[sh_state_idx + i];    

        for (int i = 0; i < action_size; i++)
            action[action_idx + i] = sh_action[sh_action_idx + i];                             
    }

    __global__ void test(
        
    ) {
        int clip = fmaxf(2, fminf(3, 10));
        printf("%d", clip);
    }



}