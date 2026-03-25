function [X, Y_residual, Y_direct, omega_dot_base_all] = build_windows(L, P, base_estimator_handle, Nwin)
% BUILD_WINDOWS  Extract sliding-window training samples from a simulation log.
%
%   [X, Y_residual, Y_direct, omega_dot_base_all] = build_windows(L, P, base_estimator_handle, Nwin)
%
%   Inputs:
%     L                     — log struct from run_episode
%     P                     — parameter struct
%     base_estimator_handle — function handle for the base estimator (e.g., @est_eso_step)
%     Nwin                  — window length (number of time steps)
%
%   Outputs:
%     X                  — input windows (numChannels × Nwin × numSamples)
%                          channels: omega_meas(3), delta_cmd(3), delta_true(3), V(1) = 10
%     Y_residual         — target residual (3 × numSamples):
%                          omega_dot_true - omega_dot_base
%     Y_direct           — target direct (3 × numSamples): omega_dot_true
%     omega_dot_base_all — base estimator output (3 × numSamples)

    N = size(L.omega_meas, 2);
    numChannels = 10;   % 3 + 3 + 3 + 1

    % ---- Run base estimator on logged measurements ----
    omega_dot_base = zeros(3, N);
    [~, est_state] = base_estimator_handle('init', [], [], P, []);
    for k = 1:N
        [omega_dot_base(:,k), est_state] = base_estimator_handle('step', ...
            L.omega_meas(:,k), L.delta_true(:,k), P, est_state);
    end

    % ---- Build airspeed vector ----
    V_vec = P.V0 * ones(1, N);   % constant airspeed

    % ---- Extract windows ----
    numSamples = N - Nwin + 1;
    X                  = zeros(numChannels, Nwin, numSamples);
    Y_residual         = zeros(3, numSamples);
    Y_direct           = zeros(3, numSamples);
    omega_dot_base_all = zeros(3, numSamples);

    for s = 1:numSamples
        idx = s : (s + Nwin - 1);          % window indices
        k   = s + Nwin - 1;                % prediction target index

        X(:, :, s) = [L.omega_meas(:, idx);     % 3 × Nwin
                       L.delta_cmd(:, idx);       % 3 × Nwin
                       L.delta_true(:, idx);      % 3 × Nwin
                       V_vec(idx)];               % 1 × Nwin

        Y_residual(:, s)         = L.omega_dot_true(:, k) - omega_dot_base(:, k);
        Y_direct(:, s)           = L.omega_dot_true(:, k);
        omega_dot_base_all(:, s) = omega_dot_base(:, k);
    end
end
