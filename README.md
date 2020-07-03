# libCEMMPC
A tiny library for CEM(cross-entropy method) based MPC(Model Predictive Control) implemented with c++ and CUDA. Could be used for long range trajectory optimization or BVP(Boundary value problem).
## Build
```bash
cd cu_cem_mpc
mkdir build
cd build
cmake ..
```
## Visualize
There are scripts inside **scripts** folder. 

In the following examples, only starts and goals are given, the model has to find out the control signal and intermediate state at each time step.
![acrobot_origin](cu_cem_mpc/imgs/acrobot/origin.gif)
![acrobot_swingup](cu_cem_mpc/imgs/acrobot/swingup.gif)
![cartpole_origin](cu_cem_mpc/imgs/cartpole/origin.gif)
![cartpole_swingup](cu_cem_mpc/imgs/cartpole/swingup.gif)
