function L = run_episode(P, estimator_handle, omega_cmd_fun)
% RUN_EPISODE  Run one closed-loop simulation episode.
%
%   L = run_episode(P, estimator_handle, omega_cmd_fun)
%
%   Supports acceleration-level disturbance (P.dist_accel_std) and
%   time-varying B0_ctrl (P.B0_ctrl_schedule).

    % Number of time steps
    N = round(P.T / P.dt);

    % Pre-allocate logs
    L.t              = zeros(1, N);
    L.omega_true     = zeros(3, N);
    L.omega_meas     = zeros(3, N);
    L.omega_dot_true = zeros(3, N);
    L.omega_dot_est  = zeros(3, N);
    L.delta_cmd      = zeros(3, N);
    L.delta_true     = zeros(3, N);
    L.omega_cmd      = zeros(3, N);
    L.dist           = zeros(3, N);

    % Initial conditions
    S.omega = zeros(3, 1);
    S.V     = P.V0;
    delta   = zeros(3, 1);

    % Sensor bias (drawn once, held constant)
    bias = P.gyro_bias_std .* randn(3, 1);

    % Make a local copy of P for time-varying parameters
    P_local = P;

    % Initialize estimator
    [~, est_state] = estimator_handle('init', [], [], P_local, []);

    % Controller state
    ctrl = struct();

    % Disturbance filter state
    dist_state.x  = zeros(3, 1);
    dist_state.xa = zeros(3, 1);

    % Check for time-varying B0_ctrl schedule
    has_schedule = isfield(P, 'B0_ctrl_schedule');

    % ---- Main simulation loop ----
    for k = 1:N
        t = (k - 1) * P.dt;

        % Time-varying B0_ctrl (linear interpolation over episode)
        if has_schedule
            frac = (k - 1) / (N - 1);
            P_local.B0_ctrl = P.B0_ctrl_schedule.start ...
                + frac * (P.B0_ctrl_schedule.finish - P.B0_ctrl_schedule.start);
        end

        % Rate command
        omega_cmd = omega_cmd_fun(t);                               % 3×1

        % Sensor measurement
        omega_meas = sensor_step(S.omega, bias, P_local);           % 3×1

        % Disturbance (torque + acceleration-level)
        [dist, dist_accel, dist_state] = disturbance_step(t, P_local, dist_state);

        % Estimator
        [omega_dot_est, est_state] = estimator_handle('step', ...
            omega_meas, delta, P_local, est_state);                 % 3×1

        % Controller
        [delta_cmd, ctrl] = indi_controller_step(omega_meas, ...
            omega_cmd, omega_dot_est, delta, P_local, ctrl);        % 3×1

        % Actuator dynamics
        delta = actuator_step(delta, delta_cmd, P_local);           % 3×1

        % Plant dynamics (with acceleration-level disturbance)
        [S, omega_dot_true] = plant_step(S, delta, P_local, dist, dist_accel);

        % Log
        L.t(k)                = t;
        L.omega_true(:, k)    = S.omega;
        L.omega_meas(:, k)    = omega_meas;
        L.omega_dot_true(:, k) = omega_dot_true;
        L.omega_dot_est(:, k) = omega_dot_est;
        L.delta_cmd(:, k)     = delta_cmd;
        L.delta_true(:, k)    = delta;
        L.omega_cmd(:, k)     = omega_cmd;
        L.dist(:, k)          = dist;
    end
end
