#include "forward_euler.h"

/**
 * @brief Performs a forward Euler integration of the ODE system 
 *        untik the final time is reached.
 * 
 * @param sys The ODE system to solve.
 * @param dt The time step size.
 * @param yf The vector to store the final solution.
 * @return 0 if the algorithm completes successfully.
 */
inline int 
forward_euler(ode_system &sys, double dt, Eigen::VectorXd &yf)
{
  yf = sys.y0;
  Eigen::VectorXd dydt(sys.dimension);
  double t = sys.t_init;
  
  // if the current time is less than the final time, keep integrating
  while (t + dt/2 < sys.t_final)
  {
    // Compute the rhs of the ODE system and place it in dydt
    sys.f(t, yf, dydt);

    // Update the solution vector and time
    yf = yf + dt*dydt;
    t = t + dt;
  }
  return 0;
}

/**
 * @brief Performs a forward Euler integration of the ODE system 
 *        until the final time is reached and stores the solution
 *        at each time step into steps.
 * 
 * @param sys The ODE system to solve.
 * @param dt The time step size.
 * @param steps The matrix to store the solution at each time step.
 * @return 0 if the algorithm completes successfully.
 */
inline int 
forward_euler_allt(ode_system &sys, double dt, Eigen::MatrixXd &steps)
{
  Eigen::VectorXd yf(sys.dimension), dydt(sys.dimension);

  yf = sys.y0; steps.row(0) = sys.y0;
  // Temporal steps initialization
  int step = 1; double t = sys.t_init;
  while (t + dt/2 < sys.t_final)
  {
    sys.f(t, yf, dydt); yf = yf + dt*dydt;
    steps.row(step) = yf;
    t = t + dt; step = step + 1;
  }
  return 0;
}
