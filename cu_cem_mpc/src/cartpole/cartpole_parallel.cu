#include "cartpole/cartpole_parallel.h"

#define I 10
#define L 2.5
#define M 10
#define m 5
#define g 9.8
#define H 0.5

#define DT  2e-3
#define MAX_T 0.2

#define PI  3.141592654f

#define MIN_X -30
#define MAX_X 30
#define MIN_V -40
#define MAX_V 40
#define MIN_W -2
#define MAX_W 2

#define MAX_TORQE 300
#define MIN_TORQE -300

#define DIM_STATE 4
#define DIM_CONTROL 1
#define STATE_X 0
#define STATE_V 1
#define STATE_THETA 2
#define STATE_W 3
#define NOBS 7
#define OBS_PENALTY 1000.0

namespace cartpole_parallel {

    __global__ void initCurand(curandState* state, unsigned long seed) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, idx, 0, &state[idx]);
    }

    __global__ 
    void set_statistics(double* d_mean_time, const double mean_time, double* d_mean_control, const double mean_control, 
        double* d_std_control, const double std_control, double* d_std_time, const double std_time, int NT){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        unsigned int id = np * NT + nt;
        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;// 0~NT * NP
        d_mean_time[id] = mean_time;
        d_mean_control[id] = mean_control;
        d_std_control[id] = std_control;
        d_std_time[id] = std_time;
        
    }

    __global__
    void set_start_state(double* temp_state, double* start, const int NS){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;
    
        temp_state[STATE_X + id*DIM_STATE] = start[STATE_X + np* DIM_STATE];
        temp_state[STATE_V + id*DIM_STATE] = start[STATE_V + np * DIM_STATE];
        temp_state[STATE_THETA + id*DIM_STATE] = start[STATE_THETA + np * DIM_STATE];
        temp_state[STATE_W + id*DIM_STATE] = start[STATE_W + np * DIM_STATE]; 
        //printf("%d: %f, %f, %f, %f\n", id, temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W]);

    }

    __global__ 
    void sampling(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time, const int NP, const int NS, const int NT, bool* active_mask,
        curandState* state){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        unsigned int id = np * NS * NT + ns * NT + nt;

        //printf("%d, %d, %d\n",np, ns, nt);

        active_mask[np * NS + ns] = true;

        double c = std_control[np * NT + nt] * curand_normal(&state[id]) + mean_control[np * NT + nt];
        if (c > MAX_TORQE) {
            c = MAX_TORQE;
        }
        else if (c < MIN_TORQE) {
            c = MIN_TORQE;
        }
        control[np * NS * NT + ns * NT + nt] = c;
        double t = std_time[np * NT + nt] * curand_normal(&state[id]) + mean_time[np * NT + nt];
        if(t < DT){
            t = 0;
        } else if (t > MAX_T) {
            t = MAX_T;
        }
        time[np * NS * NT + ns * NT + nt] = t;      
        //printf("%f, %f\n", c, t);

    }

    __device__
    bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
    // compute whether two lines intersect with each other
    {
        // ref: http://www.jeffreythompson.org/collision-detection/line-rect.php
        // calculate the direction of the lines
        double uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
        double uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    
        // if uA and uB are between 0-1, lines are colliding
        if (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1)
        {
            // intersect
            return true;
        }
        // not intersect
        return false;
    }

    __device__
    bool valid_state(double* temp_state, double* obs_list)
    {
    // check the pole with the rectangle to see if in collision
    // calculate the pole state
    // check if the position is within bound
        if (temp_state[0] < MIN_X || temp_state[0] > MAX_X)
        {
            return false;
        }
        double pole_x1 = temp_state[0];
        double pole_y1 = H;
        double pole_x2 = temp_state[0] + L * sin(temp_state[2]);
        double pole_y2 = H + L * cos(temp_state[2]);
        //std::cout << "state:" << temp_state[0] << "\n";
        //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
        //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
        for(unsigned int i = 0; i < NOBS; i++)
        {
            // check if any obstacle has intersection with pole
            //std::cout << "obstacle " << i << "\n";
            //std::cout << "points: \n";
            for (unsigned int j = 0; j < 8; j += 2)
            {
                // check each line of the obstacle
                double x1 = obs_list[i * 8 + j];
                double y1 = obs_list[i * 8 + j + 1];
                double x2 = obs_list[i * 8 + (j+2) % 8];
                double y2 = obs_list[i * 8 +(j+3) % 8];
                if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
                {
                    // intersect
                    return false;
                }
            }
        }
        return true;
    }

    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, 
        const int t_step, const int NS, const int NT, bool* active_mask, double* obs_list){
            unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int id = np * NS + ns;
            //printf("%d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
                //printf("%d\n", id);

            double t = time[np * NS * NT + ns * NT + t_step];
            if (t < 0){
                t = 0;
            }
            int num_step = t / DT;
            double _a = control[np * NS * NT + ns * NT + t_step];
                
            for(unsigned int i = 0; i < num_step; i++){
                if(!active_mask[id]){
                    break;
                }
                // update derivs
                double _v = temp_state[STATE_V + id*DIM_STATE];
                double _w = temp_state[STATE_W + id*DIM_STATE];
                double _theta = temp_state[STATE_THETA + id*DIM_STATE];
                double mass_term = 1.0 / ((M + m)*(I + m * L * L) - m * m * L * L * cos(_theta) * cos(_theta));
                deriv[STATE_X + id*DIM_STATE] = _v;
                deriv[STATE_THETA + id*DIM_STATE] = _w;
                deriv[STATE_V + id*DIM_STATE] = ((I + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) + m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term;
                deriv[STATE_W + id*DIM_STATE] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w * sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term;
                // update states
                temp_state[STATE_X + id*DIM_STATE] += DT * deriv[STATE_X + id*DIM_STATE];
                temp_state[STATE_THETA + id*DIM_STATE] += DT * deriv[STATE_THETA + id*DIM_STATE];
                temp_state[STATE_V + id*DIM_STATE] += DT * deriv[STATE_V + id*DIM_STATE];
                temp_state[STATE_W + id*DIM_STATE] += DT * deriv[STATE_W + id*DIM_STATE];
                // enforce bounds
                if (temp_state[STATE_THETA + id*DIM_STATE] > PI){
                    temp_state[STATE_THETA + id*DIM_STATE] -= 2 * PI;
                }else if(temp_state[STATE_THETA + id*DIM_STATE] < -PI){
                    temp_state[STATE_THETA + id*DIM_STATE] += 2 * PI;
                }
                if (temp_state[STATE_V + id*DIM_STATE] > MAX_V){
                    temp_state[STATE_V + id*DIM_STATE] = MAX_V;
                }else if(temp_state[STATE_V + id*DIM_STATE] < MIN_V){
                    temp_state[STATE_V + id*DIM_STATE] = MIN_V;
                }
                if (temp_state[STATE_W + id*DIM_STATE] > MAX_W){
                    temp_state[STATE_W + id*DIM_STATE] = MAX_W;
                }else if(temp_state[STATE_W + id*DIM_STATE] < MIN_W){
                    temp_state[STATE_W + id*DIM_STATE] = MIN_W;
                }
                // validate_states
                bool valid = valid_state(&temp_state[id*DIM_STATE], obs_list);
                active_mask[id] = active_mask[id] && valid;
            }        
           // printf("%d, %d: %f, %f, %f, %f\n", ns, np, temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W]);

    }

    __global__
    void get_loss(double* temp_state, double* loss, const int NS, double* goal_state, bool* active_mask){
        //printf("%d\n", id);
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;

        loss[id] = sqrt((temp_state[id * DIM_STATE + STATE_X] - goal_state[np * DIM_STATE + STATE_X]) * (temp_state[id * DIM_STATE + STATE_X] - goal_state[np * DIM_STATE + STATE_X])\
            + 0.5 * (temp_state[id * DIM_STATE + STATE_V] - goal_state[np * DIM_STATE + STATE_V]) * (temp_state[id * DIM_STATE + STATE_V] - goal_state[np * DIM_STATE + STATE_V])\
            + (temp_state[id * DIM_STATE + STATE_THETA] - goal_state[np * DIM_STATE + STATE_THETA]) * (temp_state[id * DIM_STATE + STATE_THETA] - goal_state[np * DIM_STATE + STATE_THETA])\
            + 0.5 * (temp_state[id * DIM_STATE + STATE_W] - goal_state[np * DIM_STATE + STATE_W]) * (temp_state[id * DIM_STATE + STATE_W] - goal_state[np * DIM_STATE + STATE_W]));

        if (!active_mask[id]) {
            loss[id] += OBS_PENALTY;
        }
        /*printf("%d, %d: %f, %f, %f, %f, loss: %f\n", 
            ns, np, 
            temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W],
            loss[id]);*/

    }

    __global__
    void update_statistics(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time,
        int* loss_ind, double* loss, int NP, int NS, int NT, int N_ELITE, double* best_ut){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;

        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        double sum_control = 0., sum_time = 0., ss_control = 0., ss_time = 0.;
        for(int i = 0; i < N_ELITE; i++){
            unsigned int id = np * NS * NT + loss_ind[np * NS + i] * NT + nt;
            sum_control += control[id];
            ss_control += control[id] * control[id];
            sum_time += time[id];
            ss_time += time[id] * time[id];
        }
        // printf("%f,%f\n",ss_control, ss_time);
        unsigned int s_id = np * NT + nt;
        mean_control[s_id] = sum_control / N_ELITE;
        mean_time[s_id] = sum_time / N_ELITE;
        std_control[s_id] = sqrt(ss_control / N_ELITE - mean_control[s_id] * mean_control[s_id]);
        std_time[s_id] = sqrt(ss_time / N_ELITE - mean_time[s_id] * mean_time[s_id]);
        
        best_ut[s_id] = control[np * NS * NT + loss_ind[np * NS] * NT + nt];
        best_ut[s_id + NP * NT] = time[np * NS * NT + loss_ind[np * NS] * NT + nt];
    }
    
   /* __global__ void parallel_sort(thrust::device_ptr<int> loss_ind_ptr, thrust::device_ptr<double> loss_ptr, int NS) {
        unsigned p = threadIdx.x;
        thrust::sequence(loss_ind_ptr + NS * p, loss_ind_ptr + NS * p + NS);
        thrust::sort_by_key(loss_ptr + NS * p, loss_ptr + NS * p + NS, loss_ind_ptr + NS * p);
        
    }*/

 

    CartpoleParallel::CartpoleParallel(int np, int ns, int n_elete, int nt,  int max_it, std::vector<std::vector<double>>& _obs_list, double width)
        : NP(np), NS(ns), N_ELITE(n_elete), NT(nt), max_it(max_it){
        printf("setup...\n");
        // control and time matrix
        best_ut = (double*) malloc(NP * NT * /*time + control*/2 * sizeof(double));
        cudaMalloc(&d_best_ut, NP * NT * 2 * sizeof(double)); 
        // temp state, derivative, control, time samples
            // temp_state = (double*) malloc(NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_temp_state, NP * NS * DIM_STATE * sizeof(double)); 
        cudaMalloc(&d_deriv, NP * NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_control, NP * NS * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_time, NP * NS * NT * sizeof(double));
        // for sampling statistics
        cudaMalloc(&d_mean_time, NP * NT * sizeof(double));
        cudaMalloc(&d_mean_control, NP * NT* sizeof(double));
        cudaMalloc(&d_std_control, NP * NT * sizeof(double));
        cudaMalloc(&d_std_time, NP * NT * sizeof(double));
        // for cem
        cudaMalloc(&d_loss, NP * NS * sizeof(double));
        cudaMalloc(&d_loss_ind, NP * NS * sizeof(int));
        loss_ind = (int*) malloc(NP * NS * sizeof(int));
        memset(loss_ind, 0, NP * NS  * sizeof(int));
        
        // obstacles
        cudaMalloc(&d_obs_list, NOBS * 8 * sizeof(double));
        cudaMalloc(&d_active_mask, NP * NS * sizeof(bool));

        
        obs_list = new double[NOBS*8]();
        for(unsigned i=0; i<_obs_list.size(); i++)
		{
			// each obstacle is represented by its middle point
			// calculate the four points representing the rectangle in the order
			// UL, UR, LR, LL
			// the obstacle points are concatenated for efficient calculation
			double x = _obs_list[i][0];
			double y = _obs_list[i][1];
            //std::cout << x <<","<< y << std::endl;
			obs_list[i*8 + 0] = x - width / 2;  obs_list[i*8 + 1] = y + width / 2;
			obs_list[i*8 + 2] = x + width / 2;  obs_list[i*8 + 3] = y + width / 2;
			obs_list[i*8 + 4] = x + width / 2;  obs_list[i*8 + 5] = y - width / 2;
			obs_list[i*8 + 6] = x - width / 2;  obs_list[i*8 + 7] = y - width / 2;

        }
        cudaMemcpy(d_obs_list, obs_list, sizeof(double) * NOBS * 8, cudaMemcpyHostToDevice);
        // for multiple start
        cudaMalloc(&d_start_state, NP * DIM_STATE * sizeof(double));
        cudaMalloc(&d_goal_state, NP * DIM_STATE * sizeof(double));

        // initiate curand
        cudaMalloc((void**)&devState, np * ns * nt * sizeof(curandState));
        initCurand << <(np * ns * nt + 31) / 32, 32 >> > (devState, 42);
        
        printf("done, execution:\n");

    }

    void CartpoleParallel::cem(double* start, double* goal){
        auto begin = std::chrono::system_clock::now();
        // start and goal should be NP * DIM_STATE
        cudaMemcpy(d_start_state, start, NP * DIM_STATE * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_goal_state, goal, NP * DIM_STATE * sizeof(double), cudaMemcpyHostToDevice);
        //thrust::device_ptr<double> time_ptr(d_time);
        //thrust::device_ptr<double> control_ptr(d_control);


        dim3 grid(1, 1, 1);
        dim3 grid_s(1, NS, 1);

        dim3 block_pt(NP, 1, NT);
        dim3 block_p(NP, 1, 1);

        thrust::device_ptr<double> loss_ptr(d_loss);
        thrust::device_ptr<int> loss_ind_ptr(d_loss_ind);
        //init mean
        set_statistics<<<grid, block_pt>>>(d_mean_time, 0., d_mean_control, 0.0, d_std_control, 300, d_std_time, 5e-2, NT);

        double min_loss = 1e5;
        double tmp_min_loss = 2e5;
        auto init_end = std::chrono::system_clock::now();
        //std::cout<< "start" <<std::endl;
       

        for(unsigned int it = 0; it < max_it; it ++){
            set_start_state<<<grid_s, block_p>>>(d_temp_state, d_start_state, NS);

            sampling << <grid_s, block_pt >> > (d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time, NP, NS, NT, d_active_mask, devState);

            for(unsigned int t_step = 0; t_step < NT; t_step++){
                propagate<<<grid_s, block_p >>>(d_temp_state, d_control, d_time, d_deriv, t_step, NS, NT, d_active_mask, d_obs_list);
            }
            //std::cout<< "propagation" <<std::endl;
            get_loss<<< grid_s, block_p >>>(d_temp_state, d_loss, NS, d_goal_state, d_active_mask);
            
            //std::cout<< "get_loss" <<std::endl;
            // Continue from here
            // TODO: Loop for different NP
            //parallel_sort <<<grid, NP>>>(loss_ind_ptr, loss_ptr, NS);
            for (unsigned int p = 0; p < NP; p++) {

                thrust::sequence(loss_ind_ptr + NS * p, loss_ind_ptr + NS * p + NS);

                /*std::cout << "p=" << p << std::endl;
                thrust::copy(loss_ind_ptr + NS * p, loss_ind_ptr + NS * p + NS, std::ostream_iterator<int>(std::cout, "\t"));
                std::cout << std::endl;
                thrust::copy(loss_ptr + NS * p, loss_ptr + NS * p + NS, std::ostream_iterator<double>(std::cout, "\t"));
                */
                thrust::sort_by_key(loss_ptr + NS * p, loss_ptr + NS * p + NS, loss_ind_ptr + NS * p);

               /* std::cout << "p=" << p << std::endl;
                thrust::copy(loss_ind_ptr + NS * p, loss_ind_ptr + NS * p + NS, std::ostream_iterator<int>(std::cout, "\t"));
                std::cout << std::endl;
                thrust::copy(loss_ptr + NS * p, loss_ptr + NS * p + NS, std::ostream_iterator<double>(std::cout, "\t"));*/
            }
           /* std::cout << std::endl;

            thrust::copy(loss_ptr, loss_ptr + NS * NP, std::ostream_iterator<double>(std::cout, "\t"));*/
            update_statistics<<<grid, block_pt >>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time,
                thrust::raw_pointer_cast(loss_ind_ptr),  thrust::raw_pointer_cast(loss_ptr), NP, NS, NT, N_ELITE, d_best_ut);
            //std::cout<< "update" <<std::endl;
            for (unsigned int p = 0; p < NP; p++) {
                cudaMemcpy(&tmp_min_loss, thrust::raw_pointer_cast(loss_ptr + NS * p), sizeof(double), cudaMemcpyDeviceToHost);

                /*if (tmp_min_loss < min_loss) {
                    min_loss = tmp_min_loss;
                    cudaMemcpy(best_ut, d_best_ut, 2 * NT * sizeof(double), cudaMemcpyDeviceToHost);
                }*/
                printf("p=%d, %f,\t%f\n", p, tmp_min_loss, min_loss);
                //thrust::copy(loss_ptr + NS * p, loss_ptr + NS * p + NS, std::ostream_iterator<double>(std::cout, "\t"));

            }
            printf("\n");

            /*if(min_loss < 1e-1){
                break;
            }*/
        }
        auto done = std::chrono::system_clock::now();
        printf("done\n");

        printf("control = [");
        for(unsigned int it = 0; it < NT; it ++){
            printf("%f,", best_ut[it]);
        }
        printf("]\ntime = [");
        for(unsigned int it = 0; it < NT; it ++){
            printf("%f,", best_ut[it+NT]);
        }
        printf("]\n");


        auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(init_end-begin);
        auto duration_exec = std::chrono::duration_cast<std::chrono::microseconds>(done-init_end);
        printf("init:%f\nexec:%f\n",double(duration_init.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den,
            double(duration_exec.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    
        // return d_control;
    }

    CartpoleParallel::~CartpoleParallel(){
        cudaFree(d_temp_state);
        cudaFree(d_control);
        cudaFree(d_deriv);
        cudaFree(d_time);
        cudaFree(d_mean_time);
        cudaFree(d_mean_control);
        cudaFree(d_std_control);
        cudaFree(d_std_time);
        // free(temp_state);
    }
}