% MAIN_RUN_ONE_CASE  Run a single closed-loop episode and plot results.
%
%   Uses the backward-difference + LPF estimator as baseline.
%   Produces overlay plots of commanded vs measured rate and
%   true vs estimated angular acceleration.

rng(42);

% Add paths
addpath('sim', 'estimators', 'utils');

% Load parameters
P = init_params();

% Choose estimator
estimator = @est_backdiff_lpf_step;

% Rate command: roll sinusoid at 0.5 Hz, 20 deg/s amplitude
omega_cmd_fun = @(t) [deg2rad(20)*sin(2*pi*0.5*t); 0; 0];

% Run episode
L = run_episode(P, estimator, omega_cmd_fun);

% ---- Plots ----
axis_names = {'Roll (p)', 'Pitch (q)', 'Yaw (r)'};

figure('Name', 'Rate Tracking', 'Position', [100 100 900 600]);
for ax = 1:3
    subplot(3, 1, ax);
    plot(L.t, rad2deg(L.omega_cmd(ax,:)), 'k--', 'LineWidth', 1.2); hold on;
    plot(L.t, rad2deg(L.omega_meas(ax,:)), 'b', 'LineWidth', 0.8);
    ylabel([axis_names{ax} ' (deg/s)']);
    legend('Command', 'Measured', 'Location', 'best');
    grid on;
    if ax == 1, title('Rate Tracking'); end
end
xlabel('Time (s)');

figure('Name', 'Angular Acceleration', 'Position', [100 100 900 600]);
for ax = 1:3
    subplot(3, 1, ax);
    plot(L.t, rad2deg(L.omega_dot_true(ax,:)), 'k', 'LineWidth', 1.0); hold on;
    plot(L.t, rad2deg(L.omega_dot_est(ax,:)),  'r', 'LineWidth', 0.8);
    ylabel([axis_names{ax} ' (deg/s^2)']);
    legend('True', 'Estimated', 'Location', 'best');
    grid on;
    if ax == 1, title('Angular Acceleration Estimation'); end
end
xlabel('Time (s)');

fprintf('Episode complete. %d time steps, dt = %.4f s\n', length(L.t), P.dt);
