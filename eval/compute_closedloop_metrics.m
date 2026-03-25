function M = compute_closedloop_metrics(L, P)
% COMPUTE_CLOSEDLOOP_METRICS  Compute closed-loop tracking performance metrics.
%
%   M = compute_closedloop_metrics(L, P)
%
%   Inputs:
%     L — log struct from run_episode
%     P — parameter struct (fields: dt)
%
%   Output:
%     M — struct with fields:
%           .tracking_rmse   (3×1) tracking RMSE per axis
%           .peak_error      (3×1) peak tracking error per axis
%           .actuator_rms    (3×1) actuator rate activity per axis

    tracking_error = L.omega_cmd - L.omega_meas;      % 3×N

    M.tracking_rmse = sqrt(mean(tracking_error.^2, 2));   % 3×1
    M.peak_error    = max(abs(tracking_error), [], 2);     % 3×1

    % Actuator rate activity: RMS of actuator rate
    delta_rate = diff(L.delta_cmd, 1, 2) / P.dt;          % 3×(N-1)
    M.actuator_rms = sqrt(mean(delta_rate.^2, 2));         % 3×1
end
