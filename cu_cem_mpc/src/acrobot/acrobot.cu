#include "acrobot/acrobot.h"
#include "cuda_dep.h"


#define LENGTH 20.0
#define m 1.0
#define lc 0.5
#define lc2 0.25
#define l2 1
#define I1 0.2
#define I2 1.0
#define l 1.0
#define g 9.8

#define STATE_THETA_1 0
#define STATE_THETA_2 1
#define STATE_V_1 2
#define STATE_V_2 3
#define CONTROL_T 0

#define MIN_V_1 -6
#define MAX_V_1 6
#define MIN_V_2 -6
#define MAX_V_2 6
#define MIN_T -4
#define MAX_T 4

#define DT 2e-2
#define DT_MAX 1e3
#define PI 3.141592654f
#define DIM_STATE 4
#define DIM_CONTROL 1

namespace acrobot{

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
    void set_start_state(double* temp_state, const double a1, const double a2, const double w1, const double w2, const int N){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if (id < N){
            temp_state[STATE_THETA_1 + id*DIM_STATE] = a1;
            temp_state[STATE_THETA_2 + id*DIM_STATE] = a2;
            temp_state[STATE_V_1 + id*DIM_STATE] = w1;
            temp_state[STATE_V_2 + id*DIM_STATE] = w2;
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
                    time[t * NS + id] = DT;
                }
                    // else if (time[t * NS + id] > DT_MAX){
                //     time[t * NS + id] = DT_MAX;
                // }


                if(control[t * NS + id] > MAX_T){
                    control[t * NS + id] = MAX_T;
                } else if(control[t * NS + id] < MIN_T){
                    control[t * NS + id] = MIN_T;
                }          
            }
            
        }
    }

    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, const int t_step, const int N){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if (id < N){
            double t = time[t_step*N + id];
            if (t <= DT){
                t = DT;
            }
            int num_step = t / DT;
            double _tau = control[id + t_step * N];

            if(_tau > MAX_T){
                _tau = MAX_T;
            } else if (_tau < MIN_T){
                _tau = MIN_T;
            }

            for(unsigned int i = 0; i < num_step; i++){
                // update derivs
                double theta2 = temp_state[STATE_THETA_2 + id*DIM_STATE];
                double theta1 = temp_state[STATE_THETA_1 + id*DIM_STATE] - PI / 2;
                double theta1dot = temp_state[STATE_V_1 + id*DIM_STATE];
                double theta2dot = temp_state[STATE_V_2 + id*DIM_STATE];

                //extra term m*lc2
                double d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2;

                double d22 = m * lc2 + I2;
                double d12 = m * (lc2 + l * lc * cos(theta2)) + I2;
                double d21 = d12;

                //extra theta1dot
                double c1 = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2));
                double c2 = m * l * lc * theta1dot * theta1dot * sin(theta2);
                double g1 = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2));
                double g2 = m * lc * g * cos(theta1 + theta2);

                deriv[STATE_THETA_1 + id*DIM_STATE] = theta1dot;
                deriv[STATE_THETA_2 + id*DIM_STATE] = theta2dot;
            
                double u2 = _tau - 1 * .1 * theta2dot;
                double u1 = -1 * .1 * theta1dot;
                double theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21);
                double theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21);
            
                deriv[STATE_V_1 + id*DIM_STATE] = theta1dot_dot;
                deriv[STATE_V_2 + id*DIM_STATE] = theta2dot_dot;

                temp_state[STATE_THETA_1 + id*DIM_STATE] += DT * deriv[STATE_THETA_1 + id*DIM_STATE];
                temp_state[STATE_THETA_2 + id*DIM_STATE] += DT * deriv[STATE_THETA_2 + id*DIM_STATE];
                temp_state[STATE_V_1 + id*DIM_STATE] += DT * deriv[STATE_V_1 + id*DIM_STATE];
                temp_state[STATE_V_2 + id*DIM_STATE] += DT * deriv[STATE_V_2 + id*DIM_STATE];
               
                // enforce bounds
                if(temp_state[STATE_THETA_1 + id*DIM_STATE] < -PI)
                    temp_state[STATE_THETA_1 + id*DIM_STATE] += 2 * PI;
                else if(temp_state[STATE_THETA_1 + id*DIM_STATE] > PI)
                    temp_state[STATE_THETA_1 + id*DIM_STATE] -= 2 * PI;

                if(temp_state[STATE_THETA_2 + id*DIM_STATE]< -PI)
                    temp_state[STATE_THETA_2 + id*DIM_STATE] += 2 * PI;
                else if(temp_state[STATE_THETA_2 + id*DIM_STATE] > PI)
                    temp_state[STATE_THETA_2 + id*DIM_STATE] -= 2 * PI;

                if(temp_state[STATE_V_1 + id * DIM_STATE] < MIN_V_1)
                    temp_state[STATE_V_1 + id*DIM_STATE] = MIN_V_1;
                else if(temp_state[STATE_V_1 + id*DIM_STATE] > MAX_V_1)
                    temp_state[STATE_V_1 + id*DIM_STATE] = MAX_V_1;

                if(temp_state[STATE_V_2 + id * DIM_STATE] < MIN_V_2)
                    temp_state[STATE_V_2 + id*DIM_STATE] = MIN_V_2;
                else if(temp_state[STATE_V_2 + id*DIM_STATE] > MAX_V_2)
                    temp_state[STATE_V_2 + id*DIM_STATE] = MAX_V_2;
            }        
        }
    }

    __global__
    void get_loss(double* temp_state, double* loss, const int N, double goal0, double goal1, double goal2, double goal3){
        unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;// each id is a sample
        if (id < N){
            double e1 = (temp_state[id*DIM_STATE + STATE_THETA_1]-goal0);
            double e2 = (temp_state[id*DIM_STATE + STATE_THETA_2]-goal1);

            if(e1 > PI){
                e1 -= 2*PI;
            } else if (e1 < -PI){
                e1 += 2*PI;
            }
            if(e2 > PI){
                e2 -= 2*PI;
            } else if (e2 < -PI){
                e2 += 2*PI;
            }

            loss[id] = e1*e1 + e2*e2\
                + 0.1 *(temp_state[id*DIM_STATE + STATE_V_1]-goal2) * (temp_state[id*DIM_STATE + STATE_V_1]-goal2)\
                + 0.1 * (temp_state[id*DIM_STATE + STATE_V_2]-goal3) * (temp_state[id*DIM_STATE + STATE_V_2]-goal3);
            // printf("%f, %f, %f\n", e1, e2, loss[id]);
            // if(loss[id]<0.1){
            //     printf("%f, %f, %f %f, %f\n",temp_state[id*DIM_STATE + STATE_THETA_1],temp_state[id*DIM_STATE + STATE_THETA_2],
            //     temp_state[id*DIM_STATE + STATE_V_1], temp_state[id*DIM_STATE + STATE_V_2], loss[id]);

            // }
    

            // loss[id] = 
            //     abs(sin((temp_state[id*DIM_STATE + STATE_THETA_1]-goal0)/2))\
            //     + abs((sin(temp_state[id*DIM_STATE + STATE_THETA_2]-goal1)/2)) \
            //     + 0.1 *(temp_state[id*DIM_STATE + STATE_V_1]-goal2) * (temp_state[id*DIM_STATE + STATE_V_1]-goal2)\
            //     + 0.1 * (temp_state[id*DIM_STATE + STATE_V_2]-goal3) * (temp_state[id*DIM_STATE + STATE_V_2]-goal3);
                

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
            // printf("%f, %f\n",best_ut[id],best_ut[id+NT]);
        }
    }

    Acrobot::Acrobot(int ns, int n_elete, int nt, int block_size):NS(ns),
    N_ELITE(n_elete), NT(nt), BLOCK_SIZE(block_size){
        printf("setup...\n");
        // temp_state = (double*) malloc(NS * DIM_STATE * sizeof(double));
        best_ut = (double*) malloc(NT * 2 * sizeof(double));
        cudaMalloc(&d_best_ut, NT * 2 * sizeof(double)); 

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

    void Acrobot::cem(double* start, double* goal){
        auto begin = std::chrono::system_clock::now();
        thrust::device_ptr<double> time_ptr(d_time);
        thrust::device_ptr<double> control_ptr(d_control);
        thrust::device_ptr<double> loss_ptr(d_loss);
        thrust::device_ptr<int> loss_ind_ptr(d_loss_ind);
        //init mean
        set_statistics<<<(NT+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_mean_time, 0.2, d_mean_control, 0.0, d_std_control, MAX_T, d_std_time, 2e-1, NT);

        double min_loss = 1e5;
        double tmp_min_loss = 2e5;
        auto init_end = std::chrono::system_clock::now();
        int count = 0;
        printf("goal: %f, %f, %f, %f\n",goal[0],goal[1], goal[2], goal[3]);
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
            } else {
                count ++;
            }
            printf("%f,\t%f\n", tmp_min_loss, min_loss);

            if(min_loss < 1e-2 || count > 100){
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

    Acrobot::~Acrobot(){
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