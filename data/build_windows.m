function [X, Y_residual, Y_direct, omega_dot_base_all] = build_windows(L, P, base_estimator_handle, Nwin, stride)
% BUILD_WINDOWS  Extract sliding-window training samples from a simulation log.
%
%   [X, Y_residual, Y_direct, omega_dot_base_all] = build_windows(L, P, base_estimator_handle, Nwin)
%   [X, Y_residual, Y_direct, omega_dot_base_all] = build_windows(L, P, base_estimator_handle, Nwin, stride)
%
%   Inputs:
%     L                     — log struct from run_episode
%     P                     — parameter struct
%     base_estimator_handle — function handle for the base estimator (e.g., @est_eso_step)
%     Nwin                  — window length (number of time steps)
%     stride                — (optional, default 1) step between consecutive windows.
%                             Use stride > 1 to reduce number of samples.
%
%   Outputs:
%     X                  — input windows (numChannels × Nwin × numSamples)
%                          channels: omega_meas(3), delta_cmd(3), delta_true(3), V(1) = 10
%     Y_residual         — target residual (3 × numSamples)
%     Y_direct           — target direct (3 × numSamples)
%     omega_dot_base_all — base estimator output (3 × numSamples)

    if nargin < 5, stride = 1; end

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

    % ---- Assemble full data matrix for vectorized extraction ----
    fullData = [L.omega_meas; L.delta_cmd; L.delta_true; V_vec];  % 10 × N

    % ---- Extract windows with stride ----
    starts = 1 : stride : (N - Nwin + 1);
    numSamples = length(starts);

    X                  = zeros(numChannels, Nwin, numSamples);
    Y_residual         = zeros(3, numSamples);
    Y_direct           = zeros(3, numSamples);
    omega_dot_base_all = zeros(3, numSamples);

    for si = 1:numSamples
        s   = starts(si);
        idx = s : (s + Nwin - 1);
        k   = s + Nwin - 1;

        X(:, :, si) = fullData(:, idx);

        Y_residual(:, si)         = L.omega_dot_true(:, k) - omega_dot_base(:, k);
        Y_direct(:, si)           = L.omega_dot_true(:, k);
        omega_dot_base_all(:, si) = omega_dot_base(:, k);
    end
end
