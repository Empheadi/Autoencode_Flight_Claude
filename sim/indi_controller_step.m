function [delta_cmd, ctrl] = indi_controller_step(omega_meas, omega_cmd, ...
                                   omega_dot_est, delta_prev, P, ctrl)
% INDI_CONTROLLER_STEP  One step of the INDI rate-loop controller.
%
%   [delta_cmd, ctrl] = indi_controller_step(omega_meas, omega_cmd, ...
%                                omega_dot_est, delta_prev, P, ctrl)
%
%   Inputs:
%     omega_meas    — measured angular rate (3×1, rad/s)
%     omega_cmd     — commanded angular rate (3×1, rad/s)
%     omega_dot_est — estimated angular acceleration (3×1, rad/s^2)
%     delta_prev    — previous actuator command (3×1, rad)
%     P             — parameter struct (fields: K_rate, B0_ctrl, delta_max)
%     ctrl          — controller state struct (currently unused, reserved)
%
%   Outputs:
%     delta_cmd — new actuator command (3×1, rad)
%     ctrl      — updated controller state

    % Desired angular acceleration
    omega_dot_des = P.K_rate * (omega_cmd - omega_meas);   % 3×1

    % Incremental command
    delta_inc = P.B0_ctrl \ (omega_dot_des - omega_dot_est);  % 3×1

    % Accumulate on previous command
    delta_cmd = delta_prev + delta_inc;                    % 3×1

    % Saturate to position limits
    delta_cmd = max(min(delta_cmd, P.delta_max), -P.delta_max);
end
