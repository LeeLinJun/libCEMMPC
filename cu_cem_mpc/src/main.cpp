#include "cartpole/cartpole.h"
#include "acrobot/acrobot.h"

using namespace std;

int main(){
    double start[4] = {-21.55734286,  -1.6512908 ,  -2.85039622,  -0.36047371};
    double goal[4] = {-21.87288909,  -4.25320562,  -2.92477117,  -1.03945387};
    // cartpole::Cartpole model(8192, 32, 64, 32);

    std::vector<std::vector<double>> _obs_list(
        {{-11.76184125,  -2.75},   
         { 2.07518013,  -2.75},      
         {16.75439302,   3.75},      
         { 8.07880454,   3.75},      
         {-9.24174079,   3.75},      
         {15.30796802,  -2.75},      
         { 0.64608927,   3.75}});
    cartpole::Cartpole model(8192, 32, 5, 10, 32, _obs_list, 4);
    //double start[4] = {-20,  0,  0,  0};
    //double goal[4] = {0, 0, 3.14, 0};

    // acrobot::Acrobot model(8192, 32, 10, 32);

    // double start[4] = {0, 0, 0, 0};
    // double goal[4] = {acrobot::PI, acrobot::PI ,  0,  0};
    // double goal[4] = {acrobot::PI,  .50014495,  2.63842629, -2.81358707};
    
    model.cem(start, goal);
    return 0;
}