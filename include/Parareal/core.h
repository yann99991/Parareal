#ifndef CORE_H_INCLUDED
#define CORE_H_INCLUDED

#include <Eigen/Dense>
#include <functional>

// Forward declarations to avoid circular dependencies and allow to use pointers or references to these classes in other parts
// of the code before they are fully defined.
class ode_system; class time_stepper;

// Functional typedefs

// Functions describing ode system.
// ode_rhs is now an alias for a function that takes a double and two references to Eigen::VectorXd objects, and returns an int
typedef std::function<int(double,Eigen::VectorXd &,Eigen::VectorXd &)> ode_rhs;
typedef std::function<int(double,Eigen::VectorXd &,Eigen::MatrixXd &)> ode_jac;

// Functions describing temporal integrator.
typedef std::function<int(ode_system&,double,Eigen::VectorXd &)> ode_intg;
typedef std::function<int(ode_system&,double,Eigen::MatrixXd &)> ode_intg_allt;



/**
 * @brief Represents an ordinary differential equation (ODE) system.
 */
class ode_system
{
  public:
    int dimension; /**< The dimension of the ODE system. */
    double t_init; /**< The initial time of the ODE system. */
    double t_final; /**< The final time of the ODE system. */
    Eigen::VectorXd y0; /**< The initial state vector of the ODE system. */
    ode_rhs f; /**< The right-hand side function of the ODE system. */
    ode_jac J; /**< The Jacobian matrix function of the ODE system. */

    /**
     * @brief Calculates the number of steps required to reach the final time with a given time step size.
     * @param dt The time step size.
     * @return The number of steps required.
     */
    int num_steps(double dt)
    {
      return 1 + (int) ceil( (t_final-t_init)/dt - 1/2);
    }
};


/**
 * @brief Represents a time stepper for numerical integration of ordinary differential equations.
 */
class time_stepper
{
  public:
    double dt; /**< The time step size. */
    ode_intg F; /**< Function object for integrating a single time step. */
    ode_intg_allt F_allt; /**< Function object for integrating multiple time steps. */

    /**
     * @brief Integrates the given ODE system for a single time step.
     * @param sys The ODE system to integrate.
     * @param yf The solution vector at the end of the time step.
     * @return 0 if successful, non-zero otherwise.
     */
    int integrate(ode_system &sys, Eigen::VectorXd &yf)
    {
      return F(sys, dt, yf);
    }

    /**
     * @brief Integrates the given ODE system for multiple time steps.
     * @param sys The ODE system to integrate.
     * @param steps The matrix to store the solution at each time step.
     * @return 0 if successful, non-zero otherwise.
     */
    int integrate_allt(ode_system &sys, Eigen::MatrixXd &steps)
    {
      F_allt(sys, dt, steps);
      return 0;
    }
};

#endif
