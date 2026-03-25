function [S_next, omega_dot_true] = plant_step(S, delta_actual, P, dist)
% PLANT_STEP  Advance the rigid-body plant by one time step (forward Euler).
%
%   [S_next, omega_dot_true] = plant_step(S, delta_actual, P, dist)
%
%   Inputs:
%     S             — state struct with fields:
%                       .omega  (3×1) angular rate (rad/s)
%                       .V      (scalar) airspeed (m/s)
%     delta_actual  — actual actuator deflections (3×1, rad)
%     P             — parameter struct (from init_params)
%     dist          — external disturbance torque (3×1, Nm)
%
%   Outputs:
%     S_next          — updated state struct
%     omega_dot_true  — true angular acceleration (3×1, rad/s^2)

    % Speed-scheduled aero matrices
    V = S.V;
    A = P.A0 + P.AV * (V - P.V0);           % 3×3
    B = P.B0 + P.BV * (V - P.V0);           % 3×3

    % Aerodynamic moment
    M_aero = A * S.omega + B * delta_actual; % 3×1

    % Nonlinear (cubic) damping
    M_nl = -P.mu_nl * (norm(S.omega)^2) * S.omega;  % 3×1

    % Gyroscopic torque:  omega × (I * omega)
    M_gyro = cross(S.omega, P.I * S.omega);  % 3×1

    % Total torque
    M_total = M_aero + M_nl + dist - M_gyro; % 3×1

    % Angular acceleration  (I * omega_dot = M_total)
    omega_dot_true = P.I \ M_total;          % 3×1

    % Forward Euler integration
    S_next.omega = S.omega + P.dt * omega_dot_true;  % 3×1
    S_next.V     = V;                        % constant airspeed
end
