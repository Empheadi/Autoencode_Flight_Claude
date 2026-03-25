function [S_next, omega_dot_true] = plant_step(S, delta_actual, P, dist, dist_accel)
% PLANT_STEP  Advance the rigid-body plant by one time step (forward Euler).
%
%   [S_next, omega_dot_true] = plant_step(S, delta_actual, P, dist, dist_accel)
%
%   A0 and B0 are acceleration-level matrices (rad/s^2 units), so the aero
%   contribution is computed directly as omega_dot_aero = A*omega + B*delta.
%   Gyroscopic torque and disturbance torque (Nm) are divided by I.
%   dist_accel is added directly (already in rad/s^2).
%
%   Inputs:
%     S             — state struct with fields:
%                       .omega  (3×1) angular rate (rad/s)
%                       .V      (scalar) airspeed (m/s)
%     delta_actual  — actual actuator deflections (3×1, rad)
%     P             — parameter struct (from init_params)
%     dist          — external disturbance torque (3×1, Nm)
%     dist_accel    — (optional) acceleration-level disturbance (3×1, rad/s^2)
%
%   Outputs:
%     S_next          — updated state struct
%     omega_dot_true  — true angular acceleration (3×1, rad/s^2)

    % Speed-scheduled aero matrices (acceleration-level)
    V = S.V;
    A = P.A0 + P.AV * (V - P.V0);           % 3×3, (1/s)
    B = P.B0 + P.BV * (V - P.V0);           % 3×3, (rad/s^2 / rad)

    % Aerodynamic acceleration (A0, B0 already in rad/s^2 units)
    omega_dot_aero = A * S.omega + B * delta_actual;   % 3×1, rad/s^2

    % Nonlinear (cubic) damping — acceleration level
    omega_dot_nl = -P.mu_nl * (norm(S.omega)^2) * S.omega;  % 3×1

    % Gyroscopic torque → acceleration:  I\(omega × (I*omega))
    omega_dot_gyro = P.I \ cross(S.omega, P.I * S.omega);   % 3×1

    % Disturbance torque (Nm) → acceleration
    omega_dot_dist = P.I \ dist;                             % 3×1

    % Acceleration-level disturbance (bypasses inertia)
    if nargin >= 5 && ~isempty(dist_accel)
        omega_dot_dist = omega_dot_dist + dist_accel;        % 3×1
    end

    % Total angular acceleration
    omega_dot_true = omega_dot_aero + omega_dot_nl ...
                   + omega_dot_dist - omega_dot_gyro;        % 3×1

    % Forward Euler integration
    S_next.omega = S.omega + P.dt * omega_dot_true;  % 3×1
    S_next.V     = V;                                % constant airspeed
end
