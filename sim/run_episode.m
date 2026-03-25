function L = run_episode(P, estimator_handle, omega_cmd_fun)
% RUN_EPISODE  Run one closed-loop simulation episode.
%
%   L = run_episode(P, estimator_handle, omega_cmd_fun)
%
%   Inputs:
%     P                — parameter struct (from init_params)
%     estimator_handle — function handle with signature:
%                          [omega_dot_est, est] = f(mode, omega_meas, delta, P, est)
%                        mode is 'init' or 'step'
%     omega_cmd_fun    — function handle: omega_cmd = f(t), returns 3×1 (rad/s)
%
%   Output:
%     L — log struct with fields (all time-series stored column-wise):
%           t               1×N   time (s)
%           omega_true      3×N   true angular rate
%           omega_meas      3×N   measured angular rate
%           omega_dot_true  3×N   true angular acceleration
%           omega_dot_est   3×N   estimated angular acceleration
%           delta_cmd       3×N   commanded actuator
%           delta_true      3×N   actual actuator
%           omega_cmd       3×N   rate command
%           dist            3×N   disturbance torque

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

    % Initialize estimator
    [~, est_state] = estimator_handle('init', [], [], P, []);

    % Controller state
    ctrl = struct();

    % Disturbance filter state
    dist_state.x = zeros(3, 1);

    % ---- Main simulation loop ----
    for k = 1:N
        t = (k - 1) * P.dt;

        % Rate command
        omega_cmd = omega_cmd_fun(t);                               % 3×1

        % Sensor measurement
        omega_meas = sensor_step(S.omega, bias, P);                 % 3×1

        % Disturbance
        [dist, dist_state] = disturbance_step(t, P, dist_state);    % 3×1

        % Estimator
        [omega_dot_est, est_state] = estimator_handle('step', ...
            omega_meas, delta, P, est_state);                       % 3×1

        % Controller
        [delta_cmd, ctrl] = indi_controller_step(omega_meas, ...
            omega_cmd, omega_dot_est, delta, P, ctrl);              % 3×1

        % Actuator dynamics
        delta = actuator_step(delta, delta_cmd, P);                 % 3×1

        % Plant dynamics
        [S, omega_dot_true] = plant_step(S, delta, P, dist);       % 3×1

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
