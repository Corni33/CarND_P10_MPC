#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;
  const double Lf = 2.67;
  const int desired_mpc_cycle_time_ms = 0.15;

  h.onMessage([&mpc, &Lf, &desired_mpc_cycle_time_ms](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {

    auto t1 = std::chrono::high_resolution_clock::now();

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {          

          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double x = j[1]["x"];
          double y = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          v *= 0.44704; // convert to m/s
          double delta = j[1]["steering_angle"];
          double throttle = j[1]["throttle"];  

          //predict the state of the vehicle after the delay time
          double dt = 0.1 + 0.02; // 100 ms actuator delay - 20 ms typical MPC cycle time
 
          x += v * dt * cos(psi);
          y += v * dt * sin(psi);
          psi += v * (-delta) / Lf * dt;
          v += dt * throttle;

          //transform waypoints into vehicle coordinates 
          double sin_veh = sin(psi);
          double cos_veh = cos(psi);
          for (int i = 0; i < ptsx.size(); ++i)
          {   
            // translate to vehicle position
            double x_veh = ptsx[i] - x;
            double y_veh = ptsy[i] - y;

            //rotate into vehicle coordinate system
            ptsx[i] = cos_veh*x_veh + sin_veh*y_veh;
            ptsy[i] = -sin_veh*x_veh + cos_veh*y_veh;
          }

          // fit polynomial to waypoints         
          Eigen::Map<Eigen::VectorXd> ptsx_eigen(&ptsx[0], ptsx.size());
          Eigen::Map<Eigen::VectorXd> ptsy_eigen(&ptsy[0], ptsy.size());
          
          auto coeffs = polyfit(ptsx_eigen, ptsy_eigen, 3);

          // evaluate cte and epsi
          double cte = coeffs[0]; // approximation, as cte is actually the shortest distance to the polynomial
          double epsi = -atan(coeffs[1]);

          // build the state vector
          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi;

          // call MPC solver
          auto solution_vector = mpc.Solve(state, coeffs);

          double steer_value = solution_vector[0];
          double throttle_value = solution_vector[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = rad2deg(steer_value) / 25;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          int N = 8;
          for (int i = 0; i < N; ++i)
          {
            mpc_x_vals.push_back(solution_vector[2 + i]);
            mpc_y_vals.push_back(solution_vector[2 + N + i]);
          }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for (int i = 2; i < 50; i+=2)
          {
            next_x_vals.push_back(i);
            next_y_vals.push_back(polyeval(coeffs, i));
          }         

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;


          auto t2 = std::chrono::high_resolution_clock::now();

          auto duration = std::chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

          cout << "duration in ms: " << duration << std::endl;
          
          /*if (duration < desired_mpc_cycle_time_ms)
          {
            this_thread::sleep_for(chrono::milliseconds(desired_mpc_cycle_time_ms - duration));
          }*/

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100)); 

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
