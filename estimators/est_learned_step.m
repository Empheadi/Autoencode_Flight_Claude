function [omega_dot_est, est] = est_learned_step(mode, omega_meas, delta, P, est)
% EST_LEARNED_STEP  Learned (GRU residual) estimator wrapper for closed-loop use.
%
%   [omega_dot_est, est] = est_learned_step(mode, omega_meas, delta, P, est)
%
%   Uses a trained GRU network to predict the ESO's estimation error,
%   then corrects the ESO output:  omega_dot_est = omega_dot_eso + Delta.
%
%   Inputs:
%     mode       — 'init' or 'step'
%     omega_meas — measured angular rate (3×1, rad/s)
%     delta      — actuator positions (3×1, rad)
%     P          — parameter struct (fields: V0, Nwin)
%     est        — estimator state struct
%
%   Outputs:
%     omega_dot_est — estimated angular acceleration (3×1, rad/s^2)
%     est           — updated estimator state

    if strcmp(mode, 'init')
        est.buffer_omega = zeros(3, P.Nwin);
        est.buffer_dcmd  = zeros(3, P.Nwin);
        est.buffer_dact  = zeros(3, P.Nwin);
        est.buffer_V     = zeros(1, P.Nwin);
        est.k            = 0;

        % Load trained model
        m = load('models/gru_residual_trained.mat', 'net', 'norm_stats');
        est.net        = m.net;
        est.norm_stats = m.norm_stats;

        % Initialize base ESO estimator
        [~, est.eso] = est_eso_step('init', [], [], P, []);

        omega_dot_est = zeros(3, 1);
        return
    end

    % Update buffers (shift left, append new)
    est.buffer_omega = [est.buffer_omega(:, 2:end), omega_meas];
    est.buffer_dcmd  = [est.buffer_dcmd(:, 2:end),  delta];
    est.buffer_dact  = [est.buffer_dact(:, 2:end),  delta];
    est.buffer_V     = [est.buffer_V(2:end),         P.V0];
    est.k = est.k + 1;

    % Base ESO estimate
    [omega_dot_base, est.eso] = est_eso_step('step', omega_meas, delta, P, est.eso);

    % Not enough history — fall back to ESO
    if est.k < P.Nwin
        omega_dot_est = omega_dot_base;
        return
    end

    % Build input window: 10 × Nwin
    X = [est.buffer_omega;    % 3 × Nwin
         est.buffer_dcmd;     % 3 × Nwin
         est.buffer_dact;     % 3 × Nwin
         est.buffer_V];       % 1 × Nwin

    % Normalize
    X_norm = (X - est.norm_stats.mu) ./ est.norm_stats.sigma;

    % Predict residual
    Delta = predict(est.net, dlarray(X_norm, 'CT'));  % 3×1
    Delta = extractdata(Delta);

    % Corrected estimate
    omega_dot_est = omega_dot_base + Delta;   % 3×1
end
