#include "cartpole/cartpole.h"
#include "cuda_dep.h"

#define I 10
#define L 2.5
#define M 10
#define m 5
#define g 9.8

#define DT  2e-3
#define PI  3.141592654f

#define MIN_X -30
#define MAX_X 30
#define MIN_V -40
#define MAX_V 40
#define MIN_W -2
#define MAX_W 2

#define DIM_STATE 4
#define DIM_CONTROL 1
#define STATE_X 0
#define STATE_V 1
#define STATE_THETA 2
#define STATE_W 3

namespace cartpole{

    __global__ 
    void set_statistics(double* d_mean_time, const double mean_time, double* d_mean_control, const double mean_control, 
        double* d_std_control, const double std_control, double* d_std_time, const double std_time, int N){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;// 0~NT
        if (id < N){
            d_mean_time[id] = mean_time;
            d_mean_control[id] = mean_control;
            d_std_control[id] = std_control;
            d_std_time[id] = std_time;
        }
    }

    __global__
    void set_start_state(double* temp_state, const double x, const double v, const double theta, const double w, const int N){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if (id < N){
            temp_state[STATE_X + id*DIM_STATE] = x;
            temp_state[STATE_V + id*DIM_STATE] = v;
            temp_state[STATE_THETA + id*DIM_STATE] = theta;
            temp_state[STATE_W + id*DIM_STATE] = w;
        }
    }

    __global__ 
    void sampling(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time, const int NS, const int NT){
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;// each sample
        curandState state;
        curand_init(clock(), id, 0, &state);
        if(id < NS){
            for(unsigned int t = 0; t < NT; t++){
                control[t * NS + id] = std_control[t] * curand_normal(&state) + mean_control[t];
                // curand_init(clock(), id + t, 0, &state);
                time[t * NS + id] = std_time[t] * curand_normal(&state) + mean_time[t];
                // printf("%f, %f\n",control[t * NS + id],time[t * NS + id]);
                if(time[t * NS + id] < DT){
                    time[t * NS + id] = 0;
                }
            }
            
        }
    }

    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, const int t_step, const int N){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if (id < N){
            double t = time[t_step*N + id];
            if (t < 0){
                t = 0;
            }
            int num_step = (t + 0.1*DT) / DT;
            double _a = control[id + t_step * N];

            for(unsigned int i = 0; i < num_step; i++){
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
            }        
        }
    }

    __global__
    void get_loss(double* temp_state, double* loss, const int N, double goal0, double goal1, double goal2, double goal3){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;// each id is a sample
        if (id < N){
            loss[id] = sqrt((temp_state[id*DIM_STATE + STATE_X]-goal0) * (temp_state[id*DIM_STATE + STATE_X]-goal0)\
                + 0.5*(temp_state[id*DIM_STATE + STATE_V]-goal1) * (temp_state[id*DIM_STATE + STATE_V]-goal1)\
                + (temp_state[id*DIM_STATE + STATE_THETA]-goal2) * (temp_state[id*DIM_STATE + STATE_THETA]-goal2)\
                + 0.5 * (temp_state[id*DIM_STATE + STATE_W]-goal3) * (temp_state[id*DIM_STATE + STATE_W]-goal3));
        }
    }

    __global__
    void update_statistics(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time,
        int* loss_ind, double* loss, int NS, int NT, int N_ELITE, double* best_ut){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if(id < NT){
            double sum_control = 0., sum_time = 0., ss_control = 0., ss_time = 0.;
            for(int i = 0; i < N_ELITE; i++){
                sum_control += control[loss_ind[i] + id*NS];
                ss_control += control[loss_ind[i] + id*NS] * control[loss_ind[i] + id*NS];
                sum_time += time[loss_ind[i] + id*NS];
                ss_time += time[loss_ind[i] + id*NS] * time[loss_ind[i] + id*NS];
            }
            // printf("%f,%f\n",ss_control, ss_time);
            mean_control[id] = sum_control / N_ELITE;
            mean_time[id] = sum_time / N_ELITE;
            std_control[id] = sqrt(ss_control / N_ELITE - mean_control[id] * mean_control[id]);
            std_time[id] = sqrt(ss_time / N_ELITE - mean_time[id] * mean_time[id]);
            best_ut[id] = control[loss_ind[0] + id*NS];
            best_ut[id + NT] = time[loss_ind[0] + id*NS];
            
        }
    }

    Cartpole::Cartpole(int ns, int n_elete, int nt, int block_size):NS(ns),
    N_ELITE(n_elete), NT(nt), BLOCK_SIZE(block_size){
        printf("setup...\n");
        best_ut = (double*) malloc(NT * 2 * sizeof(double));
        cudaMalloc(&d_best_ut, NT * 2 * sizeof(double)); 
        // temp_state = (double*) malloc(NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_temp_state, NS * DIM_STATE * sizeof(double)); 
        cudaMalloc(&d_deriv, NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_control, NS * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_time, NS * NT * sizeof(double));
        // for sampling
        cudaMalloc(&d_mean_time, NT * sizeof(double)); 
        cudaMalloc(&d_mean_control, NT* sizeof(double));
        cudaMalloc(&d_std_control, NT * sizeof(double));
        cudaMalloc(&d_std_time, NT * sizeof(double));
        // for cem
        cudaMalloc(&d_loss, NS * sizeof(double)); 
        cudaMalloc(&d_loss_ind, NS * sizeof(int)); 
        loss_ind = (int*) malloc(NS * sizeof(int));
        memset(loss_ind, 0, NS  * sizeof(int));
        printf("done, execution:\n");

    }

    void Cartpole::cem(double* start, double* goal){
        auto begin = std::chrono::system_clock::now();
        thrust::device_ptr<double> time_ptr(d_time);
        thrust::device_ptr<double> control_ptr(d_control);
        thrust::device_ptr<double> loss_ptr(d_loss);
        thrust::device_ptr<int> loss_ind_ptr(d_loss_ind);
        //init mean
        set_statistics<<<(NT+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_mean_time, 0, d_mean_control, 0.0, d_std_control, 500, d_std_time, 5e-2, NT);

        double min_loss = 1e5;
        double tmp_min_loss = 2e5;
        auto init_end = std::chrono::system_clock::now();

        for(unsigned int it = 0; it < 100; it ++){
            set_start_state<<<(NS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_temp_state, start[0], start[1], start[2], start[3], NS);

            sampling<<<(NS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time, NS, NT);
            for(unsigned int t_step = 0; t_step < NT; t_step++){
                propagate<<<(NS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_temp_state, d_control, d_time, d_deriv, t_step, NS);
            }
            get_loss<<<(NS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_temp_state, d_loss, NS, goal[0], goal[1], goal[2], goal[3]);
            thrust::sequence(loss_ind_ptr, loss_ind_ptr+NS);
            thrust::sort_by_key(loss_ptr, loss_ptr + NS, loss_ind_ptr);

            update_statistics<<<NT, BLOCK_SIZE>>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time,
                thrust::raw_pointer_cast(loss_ind_ptr),  thrust::raw_pointer_cast(loss_ptr), NS, NT, N_ELITE, d_best_ut);

            cudaMemcpy(&tmp_min_loss, thrust::raw_pointer_cast(loss_ptr), sizeof(double), cudaMemcpyDeviceToHost);

            if(tmp_min_loss < min_loss){
                min_loss = tmp_min_loss;
                cudaMemcpy(best_ut, d_best_ut, 2 * NT * sizeof(double), cudaMemcpyDeviceToHost);

            }
            printf("%f,\t%f\n", tmp_min_loss, min_loss);

            if(min_loss < 1e-1){
                break;
            }
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

    Cartpole::~Cartpole(){
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