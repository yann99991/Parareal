#ifndef PARAREAL_H_INCLUDED
#define PARAREAL_H_INCLUDED

#include "core.h"

// Parareal Method. There are 2 different implementations of the Parareal method.
// Naive and pipelined.
inline int 
parareal(ode_system &, time_stepper, time_stepper, int, Eigen::MatrixXd &);

inline int 
pipelined_parareal(ode_system &, 
                   time_stepper, time_stepper,
                   int, Eigen::MatrixXd &);

#include "parareal.cpp"

#endif
