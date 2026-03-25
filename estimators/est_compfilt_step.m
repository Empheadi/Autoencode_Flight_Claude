function [omega_dot_est, est] = est_compfilt_step(mode, omega_meas, delta, P, est)
% EST_COMPFILT_STEP  Complementary filter estimator (model + derivative blend).
%
%   [omega_dot_est, est] = est_compfilt_step(mode, omega_meas, delta, P, est)
%
%   Fuses a numerical derivative (high-frequency, noisy) with a model-based
%   prediction (low-frequency, biased by model error) and applies a low-pass
%   filter to the blended result.
%
%   Inputs:
%     mode       — 'init' or 'step'
%     omega_meas — measured angular rate (3×1, rad/s)
%     delta      — actuator positions (3×1, rad)
%     P          — parameter struct
%     est        — estimator state struct
%
%   Outputs:
%     omega_dot_est — estimated angular acceleration (3×1, rad/s^2)
%     est           — updated estimator state

    if strcmp(mode, 'init')
        est.omega_prev     = zeros(3, 1);
        est.omega_dot_filt = zeros(3, 1);
        omega_dot_est = zeros(3, 1);
        return
    end

    % ---- Numerical derivative (high-frequency info, noisy) ----
    omega_dot_diff = (omega_meas - est.omega_prev) / P.dt;    % 3×1
    est.omega_prev = omega_meas;

    % ---- Model-based prediction (low-frequency info) ----
    V = P.V0;
    A = P.A0 + P.AV * (V - P.V0);                              % 3×3
    B = P.B0_ctrl + P.BV * (V - P.V0);                         % 3×3 (controller's B)

    M_model = A * omega_meas + B * delta ...
            - cross(omega_meas, P.I * omega_meas);              % 3×1
    omega_dot_model = P.I \ M_model;                            % 3×1

    % ---- Blend ----
    alpha = P.est_cf_alpha;   % 0 = pure model, 1 = pure derivative
    omega_dot_raw = alpha * omega_dot_diff + (1 - alpha) * omega_dot_model;

    % ---- Low-pass filter ----
    est.omega_dot_filt = P.est_lpf_alpha * est.omega_dot_filt ...
                       + (1 - P.est_lpf_alpha) * omega_dot_raw;

    omega_dot_est = est.omega_dot_filt;   % 3×1
end
