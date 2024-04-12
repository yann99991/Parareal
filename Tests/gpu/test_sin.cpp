#include <stdio.h>
#include <omp.h>

#include <iostream>

#include <Eigen/Dense>
#include <Parareal/core.h>
#include <Parareal/parareal.h>
#include <Parareal/forward_euler.h>

#include "rhs_sin.h"

typedef Eigen::VectorXd Evec;
typedef Eigen::MatrixXd Emat;

int main(int argc, char **argv)
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  int P = numberOfSMs;

  // ODE system setup
  ode_system ode;
  ode.dimension = 1; ode.t_init = 0; ode.t_final = 1.0*P;
  ode.y0 = Evec(1); ode.y0(0) = 1;

  std::cout << "Solving system of dimension nx = " << ode.dimension 
            << " with initial condition y0 = "     << ode.y0(0)     << std::endl;
  // rhs
  ode.f = std::function<int(double, Evec&, Evec&)>(&linear1d);

  // Time steppers setup: Both a forward Euler method but with different time steps
  time_stepper course; time_stepper fine;

  course.dt = 0.5;
  course.F = std::function<int(ode_system&, double, Evec &)>(&forward_euler);
  course.F_allt = std::function<int(ode_system&, double, Emat &)>(&forward_euler_allt);
  
  // the fine time stepper is still a forward Euler method, but with a smaller time step
  fine.dt = .0000001;
  fine.F = std::function<int(ode_system&, double, Evec &)>(&forward_euler);
  fine.F_allt = std::function<int(ode_system&, double, Emat &)>(&forward_euler_allt);

  // Test Parareal
  printf("Using %d processors to solve up to T = %f\n", P, ode.t_final);

  // Number of coarse steps
  int csteps = ode.num_steps(course.dt);
  double tt = 0.0;

  
  Emat yf(csteps, ode.dimension);

  // timestamp
  tt = omp_get_wtime();

  // Parareal algorithm
  parareal(ode, course, fine, 4, yf);
  
  // timestamp
  tt = omp_get_wtime() - tt;

  std::cout << "Solution : \n" << yf << std::endl;
  printf("Time taken %f\n", tt);

  return 0;
}
