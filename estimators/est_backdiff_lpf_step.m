function [omega_dot_est, est] = est_backdiff_lpf_step(mode, omega_meas, delta, P, est)
% EST_BACKDIFF_LPF_STEP  Backward-difference + low-pass filter estimator.
%
%   [omega_dot_est, est] = est_backdiff_lpf_step(mode, omega_meas, delta, P, est)
%
%   Inputs:
%     mode       — 'init' or 'step'
%     omega_meas — measured angular rate (3×1, rad/s)   [ignored in 'init']
%     delta      — actuator positions (3×1, rad)         [unused]
%     P          — parameter struct (fields: dt, est_lpf_alpha)
%     est        — estimator state struct
%
%   Outputs:
%     omega_dot_est — estimated angular acceleration (3×1, rad/s^2)
%     est           — updated estimator state

    if strcmp(mode, 'init')
        est.omega_prev    = zeros(3, 1);
        est.omega_dot_filt = zeros(3, 1);
        omega_dot_est = zeros(3, 1);
        return
    end

    % Backward difference
    omega_dot_raw = (omega_meas - est.omega_prev) / P.dt;   % 3×1

    % First-order low-pass filter
    est.omega_dot_filt = P.est_lpf_alpha * est.omega_dot_filt ...
                       + (1 - P.est_lpf_alpha) * omega_dot_raw;

    omega_dot_est = est.omega_dot_filt;   % 3×1
    est.omega_prev = omega_meas;
end
