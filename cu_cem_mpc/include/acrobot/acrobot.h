#pragma once
#include <chrono>   
#include <stdio.h>
// #include "cartpole_cu.h"


namespace acrobot{
   
    class Acrobot{
        public:
            int NS;
            int N_ELITE;
            int NT;
            Acrobot(int NS, int N_ELITE, int NT, int BLOCK_SIZE);
            ~Acrobot();
            void cem(double* start, double* goal);

            // double* cem(double* start, double* goal);
        private:
            int BLOCK_SIZE;
            double *temp_state, *d_temp_state, *d_control, *d_deriv, *d_time;
            double *d_mean_time, *d_mean_control, *d_std_control, *d_std_time;
            double *d_loss;
            int *d_loss_ind, *loss_ind;
            double *best_ut, *d_best_ut;
            
            
    };
}
