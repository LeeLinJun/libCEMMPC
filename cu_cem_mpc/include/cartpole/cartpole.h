#pragma once
#include <chrono>   
#include <stdio.h>
#include <vector>
// #include "cartpole_cu.h"

namespace cartpole{

    class Cartpole{
        public:
            int NS;
            int N_ELITE;
            int NT;
            int max_it;
            // Cartpole();
            Cartpole(int NS, int N_ELITE, int NT, int max_it, int BLOCK_SIZE, std::vector<std::vector<double>>& _obs_list, double width);
            ~Cartpole();
            void cem(double* start, double* goal);

            // double* cem(double* start, double* goal);
        protected:
            int BLOCK_SIZE;
            double *temp_state, *d_temp_state, *d_control, *d_deriv, *d_time;
            double *d_mean_time, *d_mean_control, *d_std_control, *d_std_time;
            double *d_loss;
            int *d_loss_ind, *loss_ind;
            double *best_ut, *d_best_ut;
            // for obstacles
            double* d_obs_list, *obs_list;
            bool* d_active_mask;
            
    };
}
