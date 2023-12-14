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
__constant__ float BALL_VELOCITY_COEF = 3;
__constant__ float BALL_BOUNCE_MULTIPLIER = 1;

__constant__ float EPISODE_LENGTH = 4000;

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

    __device__ float dis(
        float x1,
        float y1, 
        float x2,
        float y2
    ) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return sqrtf(dx * dx + dy * dy);
    }


    __device__ void set_rew(
        float* rew,
        float* state,
        int state_size,
        float* rew_map,
        int* goal_scored
    ) {
        float reward = 0;

        float* ball = &state[(blockDim.x - 1) * state_size];
        // float* agent = &state[threadIdx.x * state_size];

        // goal
        if (goal_scored[blockIdx.x] == 1) {
            reward += rew_map[0];
        }
        
        // own goal
        if (goal_scored[blockIdx.x] == -1) {
            reward -= rew_map[0];
        } 

        float prev_ball_to_goal = dis(ball[4], ball[5], 4800, 0);
        float ball_to_goal = dis(ball[0], ball[1], 4800, 0);

        reward += rew_map[1] * (prev_ball_to_goal - ball_to_goal);

        rew[0] = reward;
    }


    __device__ void set_rel_obs(
        float* obs,
        float* agent,
        float* other
    ) {
        float agent_angle = agent[2];

        float x = other[0] - agent[0];
        float y = other[1] - agent[1];
        float angle = atan2(y, x) - agent_angle;

        float x_prime = x * cosf(-agent_angle) - y * sinf(-agent_angle);
        float y_prime = x * sinf(-agent_angle) + y * cosf(-agent_angle);

        obs[0] = x_prime / 10000;
        obs[1] = y_prime / 10000;
        obs[2] = sinf(angle);
        obs[3] = cosf(angle);

    }

    __device__ void set_rel_obs_xy(
        float* obs,
        float* agent,
        float other_x,
        float other_y
    ) {
        float agent_angle = agent[2];

        float x = other_x - agent[0];
        float y = other_y - agent[1];
        float angle = atan2(y, x) - agent_angle;

        float x_prime = x * cosf(-agent_angle) - y * sinf(-agent_angle);
        float y_prime = x * sinf(-agent_angle) + y * cosf(-agent_angle);

        obs[0] = x_prime / 10000;
        obs[1] = y_prime / 10000;
        obs[2] = sinf(angle);
        obs[3] = cosf(angle);
    }

    __device__ void set_obs(
        float* obs,
        float* state,
        int state_size,
        bool is_opponent,
        int n_agents,
        int n_opponents
    ) {
        float* self = &state[threadIdx.x * state_size];

        // ball
        set_rel_obs(&obs[0], self, &state[(blockDim.x - 1) * state_size]);

        // 1-hot for can kick
        obs[4] = -1;

        if (is_opponent) {

            // opponents first
            for (int i = n_agents; i < blockDim.x - 1; i++) {
                if (i == threadIdx.x) continue;

                set_rel_obs(&obs[5 + (i - n_agents) * 4], self, &state[i * state_size]);
            }

            // then agents
            for (int i = 0; i < n_agents; i++) {
                if (i == threadIdx.x) continue;

                set_rel_obs(&obs[5 + (i + n_opponents) * 4], self, &state[i * state_size]);
            }

        } else {

            // agents first then opponents
            for (int i = 0; i < blockDim.x - 1; i++) {
                if (i == threadIdx.x) continue;

                set_rel_obs(&obs[5 + i * 4], self, &state[i * state_size]);
            }

        }

        // goal
        set_rel_obs_xy(&obs[5 + (blockDim.x - 1) * 4], self, 4800, 0);

        // opponent goal
        set_rel_obs_xy(&obs[5 + (blockDim.x - 1) * 4 + 4], self, -4800, 0);
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
        int* goal_scored,
        int seed
    ) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // update ball velocity
        float* ball = &state[(blockDim.x - 1) * state_size];

        ball[4] = ball[0];
        ball[5] = ball[1];

        ball[3] += BALL_ACCELERATION;
        ball[3] = clip(ball[3], 0, 100);

        // update ball position
        ball[0] += ball[3] * cosf(ball[2]);
        ball[1] += ball[3] * sinf(ball[2]);

        // -1 for ball
        int num_entities = blockDim.x - 1;
        for (int i = 0; i < num_entities; i++) {

            int entity_idx = i * state_size;

            // compute distance between ball and entity
            float dx = ball[0] - state[entity_idx + 0];
            float dy = ball[1] - state[entity_idx + 1];
            float distance = sqrtf(dx * dx + dy * dy);

            // skip if no collision
            if (distance >= (ROBOT_RADIUS + BALL_RADIUS) * 6) continue;

            curandState_t rng;
            curand_init(__float2int_rd(seed + tid), tid, 0, &rng);

            ball[3] = BALL_VELOCITY_COEF * 10;
            ball[2] = atan2f(dy, dx);
            ball[2] += random_normal(&rng, -1, 1) * M_PI / 8;

        }

        bool bounce = true;
        int ball_x_sign = ball[0] < 0 ? -1 : 1;
        int ball_y_sign = ball[1] < 0 ? -1 : 1;
        
        if (abs(ball[1]) > 3000) {
            if (bounce) {

                ball[1] = ball_y_sign * 3000;
                ball[3] *= BALL_BOUNCE_MULTIPLIER;
                ball[2] *= -1;

            } else {

                // TODO 

            }
        }

        if (abs(ball[0]) > 4500 && abs(ball[1]) > 1100) {
            if (bounce) {

                ball[0] = ball_x_sign * 4500;
                ball[3] *= BALL_BOUNCE_MULTIPLIER;
                ball[2] = M_PI - ball[2];

            } else {

                // TODO 

            }
        }

        // goal
        if (ball[0] > 4500 && ball[1] < 1100 && ball[1] > -1100) {
            goal_scored[blockIdx.x] = 1;
        }

        // own goal
        if (ball[0] < -4500 && ball[1] < 1100 && ball[1] > -1100) {
            goal_scored[blockIdx.x] = -1;
        }

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
        int* goal_scored,
        int num_agents,
        int num_opponents,
        float* obs,
        int seed
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int state_idx = blockIdx.x * blockDim.x * state_size + threadIdx.x * state_size;

        curandState_t rng;
        curand_init(seed + tid, tid, 0, &rng);

        if (threadIdx.x == 0) {
            goal_scored[blockDim.x] = 0;

            float* agent = &state[state_idx];

            agent[0] = 0;
            agent[1] = 0;
            agent[2] = 0;
            agent[3] = 0;
            agent[4] = 0;
            agent[5] = 0;
            agent[6] = 0;
            return;
        }

        // agents
        if (threadIdx.x < num_agents) {

            float* agent = &state[state_idx];

            agent[0] = random_uniform(&rng, -4500, 0);
            agent[1] = random_uniform(&rng, -3000, 3000);
            agent[2] = random_uniform(&rng, -M_PI, M_PI);
            agent[3] = 0;
            agent[4] = 0;
            agent[5] = 0;
            agent[6] = 0;

        // opponents
        } else if (threadIdx.x < num_agents + num_opponents) {

            float* opponent = &state[state_idx];

            opponent[0] = random_uniform(&rng, -4500, 0);
            opponent[1] = random_uniform(&rng, -3000, 3000);
            opponent[2] = random_uniform(&rng, -M_PI, M_PI);
            opponent[3] = 0;
            opponent[4] = 0;
            opponent[5] = 0;
            opponent[6] = 0;

        // ball
        } else {

            float* ball = &state[state_idx];

            ball[0] = 500;
            ball[1] = 0;
            ball[2] = 0;
            ball[3] = 0;
            ball[4] = 0;
            ball[5] = 0;

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

        if (threadIdx.x >= blockDim.x - 1) return;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int action_idx = blockIdx.x * (blockDim.x - 1) * action_size + threadIdx.x * action_size;
        curandState_t rng;
        curand_init(__float2int_rd(seed + tid), tid, 0, &rng);
        int step_size = 1;

        action[action_idx + 0] = random_uniform(&rng, -step_size, step_size);
        action[action_idx + 1] = random_uniform(&rng, -step_size, step_size);
        action[action_idx + 2] = random_uniform(&rng, -M_PI, M_PI);
        // if (threadIdx.x == 0) {
        //     action[action_idx + 0] = .5;
        //     action[action_idx + 1] = 0;
        //     action[action_idx + 2] = 0;
        //     return;
        // }

        // action[action_idx + 0] = 0;
        // action[action_idx + 1] = 0;
        // action[action_idx + 2] = 0;    

        // action[action_idx + 0] = 0;
        // action[action_idx + 1] = 0;
        // action[action_idx + 2] = 0;
        
    }

    __global__ void step2(
        float* state, 
        int state_size,
        float* action,
        int action_size,
        int obs_size,
        float* rew_map,
        int n_agents,
        int n_opponents,
        int* time,
        int seed,
        float* obs,
        float* rew,
        bool* term,
        bool* trun,
        float* info
    ) { 
        return;

    }

    __global__ void step(
        float* state, 
        int state_size,
        float* action,
        int action_size,
        int obs_size,
        float* rew_map,
        int* goal_scored,
        int n_agents,
        int n_opponents,
        int* time,
        int seed,
        float* obs,
        float* rew,
        bool* term,
        bool* trun,
        float* info
    ) {            

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // shared memory
        extern __shared__ float sh_mem[];

        // pointer to the start of state array
        float* sh_state = &sh_mem[0];

        // pointer to the start of action array
        float* sh_action = &sh_mem[blockDim.x * state_size];

        // shared memory indicies corresponding to this thread
        int sh_state_idx = threadIdx.x * state_size;
        int sh_action_idx = threadIdx.x * action_size;

        // global memory indicies corresponding to this thread
        int state_idx = blockIdx.x * blockDim.x * state_size + sh_state_idx;
        int action_idx = blockIdx.x * (blockDim.x - 1) * action_size + sh_action_idx;

        // load shared memory
        for (int i = 0; i < state_size; i++)
            sh_state[sh_state_idx + i] = state[state_idx + i];

        if (threadIdx.x < n_agents + n_opponents) {
            for (int j = 0; j < action_size; j++) {
                sh_action[sh_action_idx + j] = action[action_idx + j];
            }
        }

        // only first thread in first block updates the time
        if (tid == 0) time[0]++;  

        // ensure that all threads have finished loading shared memory before continuing
        __syncthreads();

        if (threadIdx.x < n_agents) {
            move_agent(&sh_state[sh_state_idx], &sh_action[sh_action_idx]);
        } else if (threadIdx.x < n_agents + n_opponents) {
            move_opponent(&sh_state[sh_state_idx], &sh_action[sh_action_idx]);
        } 

        // ensures that all agents/opponents have moved before checking for collisions
        // prevents race conditions (ex. detecting collision with an agent that hasn't been moved yet)
        __syncthreads();

        if (threadIdx.x < n_agents + n_opponents) {
            check_collisions(sh_state, state_size);
        } else {
            // last thread updates ball
            update_ball(sh_state, state_size, goal_scored, seed);
        }

        // ensures that all agents/opponents/ball have stopped changing
        __syncthreads();

        int obs_idx = blockIdx.x * (n_agents + n_opponents) * obs_size + threadIdx.x * obs_size;

        if (threadIdx.x < n_agents) {

            int idx = blockIdx.x * n_agents + threadIdx.x;

            set_obs(&obs[obs_idx], sh_state, state_size, false, n_agents, n_opponents);
            set_rew(&rew[idx], sh_state, state_size, rew_map, goal_scored);

            term[idx] = time[0] > EPISODE_LENGTH || goal_scored[blockIdx.x] == 1;
            trun[idx] = false;
            info[idx] = 0;

        } else if (threadIdx.x < n_agents + n_opponents) {

            set_obs(&obs[obs_idx], sh_state, state_size, true, n_agents, n_opponents);
        }

        __syncthreads();

        for (int i = 0; i < state_size; i++)
            state[state_idx + i] = sh_state[sh_state_idx + i];    

        for (int i = 0; i < action_size; i++)
            action[action_idx + i] = sh_action[sh_action_idx + i]; 

                          
    }

    __global__ void test(
        
    ) {
        // int clip = fmaxf(2, fminf(3, 10));
        // printf("%d", clip);
    }



}