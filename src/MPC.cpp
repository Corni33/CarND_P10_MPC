#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

//using CppAD::AD;

size_t N = 10;
double dt = 0.1;

// calculate start indices of variables
size_t x_start      = 0 * N;
size_t y_start      = 1 * N;
size_t psi_start    = 2 * N;
size_t v_start      = 3 * N;
size_t cte_start    = 4 * N;
size_t epsi_start   = 5 * N;
size_t delta_start  = 6 * N; 
size_t a_start      = 7 * N - 1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  double c_state = 1.0;
  double c_act = 30.0;
  double c_act_seq = 10.0;

  // TODO: adapt velocity dynamically based on curvature
  double v_target = 30;

  typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

  void operator()(ADvector& fg, const ADvector& vars) {   
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // Cost function

    // The part of the cost based on the reference state.
    for (int t = 0; t < N; t++) {
      fg[0] += c_state*CppAD::pow(vars[cte_start + t], 2);
      fg[0] += c_state*CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += c_state*CppAD::pow(vars[v_start + t] - v_target, 2);
    }

    // Minimize the use of actuators.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += c_act*CppAD::pow(vars[delta_start + t], 2);
      fg[0] += c_act*CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int t = 0; t < N - 2; t++) {
      fg[0] += c_act_seq*CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += c_act_seq*CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // -----------------------------------------------------------------

    // Initial constraints    
    fg[1 + x_start]     = vars[x_start];
    fg[1 + y_start]     = vars[y_start];
    fg[1 + psi_start]   = vars[psi_start];
    fg[1 + v_start]     = vars[v_start];
    fg[1 + cte_start]   = vars[cte_start];
    fg[1 + epsi_start]  = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; t++) {      

      // The state at time t.
      CppAD::AD<double> x0 = vars[x_start + t - 1];
      CppAD::AD<double> y0 = vars[y_start + t - 1];
      CppAD::AD<double> psi0 = vars[psi_start + t - 1];
      CppAD::AD<double> v0 = vars[v_start + t - 1];
      CppAD::AD<double> cte0 = vars[cte_start + t - 1];
      CppAD::AD<double> epsi0 = vars[epsi_start + t - 1];

      // The state at time t+1 .
      CppAD::AD<double> x1 = vars[x_start + t];
      CppAD::AD<double> y1 = vars[y_start + t];
      CppAD::AD<double> psi1 = vars[psi_start + t];
      CppAD::AD<double> v1 = vars[v_start + t];
      CppAD::AD<double> cte1 = vars[cte_start + t];
      CppAD::AD<double> epsi1 = vars[epsi_start + t];

      // Only consider the actuation at time t.
      CppAD::AD<double> delta0 = vars[delta_start + t - 1];
      CppAD::AD<double> a0 = vars[a_start + t - 1];

      //CppAD::AD<double> f0 = coeffs[0] + coeffs[1] * x0;
      //CppAD::AD<double> psides0 = CppAD::atan(coeffs[1]);

      CppAD::AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0*x0 + coeffs[3] * x0*x0*x0;
      CppAD::AD<double> psides0 = CppAD::atan(3.0 * coeffs[3] * x0 * x0 + 2.0 * coeffs[2] * x0 + coeffs[1]);

      // The idea here is to constraint this value to be 0.
      fg[1 + x_start + t]     = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t]     = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t]   = psi1 - (psi0 + v0 * (-delta0) / Lf * dt); // "-"delta0 because of unity sign convention
      fg[1 + v_start + t]     = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t]   = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t]  = epsi1 - ((psi0 - psides0) + v0 * (-delta0) / Lf * dt); // "-"delta0 because of unity sign convention

    }

  }
};



//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x    = state[0];
  double y    = state[1];
  double psi  = state[2];
  double v    = state[3];
  double cte  = state[4];
  double epsi = state[5];

  // ------------------------------------------------------------------------

  // number of variables
  // 6 states (including cte and epsi); 2 actuator inputs
  size_t n_vars = 6*N + 2*(N-1); 

  // number of constraints
  size_t n_constraints = 6 * N;

  // ------------------------------------------------------------------------

  // Initial value of the independent variables.
  // Set all to 0 except initial state
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // Set the initial variable values
  vars[x_start]     = x;
  vars[y_start]     = y;
  vars[psi_start]   = psi;
  vars[v_start]     = v;
  vars[cte_start]   = cte; 
  vars[epsi_start]  = epsi;

  // ------------------------------------------------------------------------

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.3; //max steering angle: 25 degrees (=0.436332 rad)
    vars_upperbound[i] = 0.3;
  }

  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0; // max deceleration/acceleration: (-)1
    vars_upperbound[i] = 1.0;
  }

  // ------------------------------------------------------------------------

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);

  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // ------------------------------------------------------------------------

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // return first control input values 
  std::vector<double> control_vector = { solution.x[delta_start], solution.x[a_start] };

  for (int i = 0; i < N; ++i)
  {
    control_vector.push_back(solution.x[x_start + i]);
  }
  for (int i = 0; i < N; ++i)
  {
    control_vector.push_back(solution.x[y_start + i]);
  }

  return control_vector;
}
