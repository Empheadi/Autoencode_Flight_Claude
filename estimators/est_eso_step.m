function [omega_dot_est, est] = est_eso_step(mode, omega_meas, delta, P, est)
% EST_ESO_STEP  Second-order Extended State Observer for angular acceleration.
%
%   [omega_dot_est, est] = est_eso_step(mode, omega_meas, delta, P, est)
%
%   The ESO treats omega_dot as an extended state and estimates it from
%   gyro measurements using a Luenberger-style observer with bandwidth w_o.
%
%   Inputs:
%     mode       — 'init' or 'step'
%     omega_meas — measured angular rate (3×1, rad/s)
%     delta      — actuator positions (3×1, rad)  [unused]
%     P          — parameter struct (fields: dt, est_eso_bw)
%     est        — estimator state struct
%
%   Outputs:
%     omega_dot_est — estimated angular acceleration (3×1, rad/s^2)
%     est           — updated estimator state

    if strcmp(mode, 'init')
        est.z1 = zeros(3, 1);   % rate estimate     (3×1)
        est.z2 = zeros(3, 1);   % accel estimate    (3×1)
        omega_dot_est = zeros(3, 1);
        return
    end

    % Observer gains from bandwidth
    w_o   = P.est_eso_bw;       % rad/s
    beta1 = 2 * w_o;
    beta2 = w_o^2;

    % Observation error
    e = omega_meas - est.z1;    % 3×1

    % Update estimates (forward Euler)
    est.z1 = est.z1 + P.dt * (est.z2 + beta1 * e);   % 3×1
    est.z2 = est.z2 + P.dt * (beta2 * e);             % 3×1

    omega_dot_est = est.z2;     % 3×1
end
