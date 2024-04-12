
#ifndef RHS_SIN_H
#define RHS_SIN_H

#include <Eigen/Core>
#include <cmath>
#define LAMBDA -1

// 

// VectorXd represents a vector of dynamic size. 
/**
 * @brief Calculates the rhs of the linear 1D ODE system.
 *        u' = (LAMBDA)*u
 * @param t The current time.
 * @param yf The vector containing the function values.
 * @param dydt The vector to store the calculated rhs.
 * @return int Returns 0 on success.
 */
inline int 
linear1d(double t, Eigen::VectorXd &yf, Eigen::VectorXd &dydt)
{
  // 0 is the current time index
  dydt(0) = std::sin(t);
  return 0;
}

#endif
