% MAIN_EVAL_CLOSEDLOOP  Evaluate estimators in closed-loop INDI control.
%
%   Runs ten scenarios with each estimator and compares tracking performance.
%   Scenarios 1-5: baseline conditions (no delay).
%   Scenarios 6-10: stress tests with sensor delay, mismatch, and disturbance
%     that expose the GRU's prediction advantage.

rng(42);
addpath('sim', 'estimators', 'utils', 'eval', 'models');

P = init_params();

estimators = {@est_backdiff_lpf_step, @est_eso_step, @est_compfilt_step};
est_names  = {'BackDiff', 'ESO', 'CompFilt'};
if exist('models/gru_residual_trained.mat', 'file')
    estimators{end+1} = @est_learned_step;
    est_names{end+1}  = 'GRU-Res';
end

% ---- Define scenarios ----
scenarios = struct();

% === Baseline (no delay) ===

% Scenario 1: Roll step
scenarios(1).name = 'Roll Step';
scenarios(1).cmd  = @(t) [deg2rad(25)*(t>1 & t<3); 0; 0];
scenarios(1).P    = P;

% Scenario 2: Roll chirp
scenarios(2).name = 'Roll Chirp';
scenarios(2).cmd  = @(t) [deg2rad(15)*sin(2*pi*(0.5 + (10-0.5)/(2*10)*t)*t); 0; 0];
scenarios(2).P    = P;

% Scenario 3: Multi-axis sinusoidal
scenarios(3).name = 'Multi-Axis';
scenarios(3).cmd  = @(t) [deg2rad(20)*sin(2*pi*1*t); ...
                           deg2rad(10)*sin(2*pi*2*t); ...
                           deg2rad(5)*sin(2*pi*0.5*t)];
scenarios(3).P    = P;

% Scenario 4: Disturbance rejection
P4 = P;
P4.dist_type = 'step';
P4.dist_std  = [10; 10; 5];
P4.T = 10;
scenarios(4).name = 'Dist Rejection';
scenarios(4).cmd  = @(t) [0; 0; 0];
scenarios(4).P    = P4;

% Scenario 5: B mismatch (no delay)
P5 = P;
P5.B0_ctrl = P.B0 * 0.8;
scenarios(5).name = 'B Mismatch';
scenarios(5).cmd  = @(t) [deg2rad(25)*(t>1 & t<3); 0; 0];
scenarios(5).P    = P5;

% === Stress tests (sensor delay + other challenges) ===

% Scenario 6: Sensor delay only (2 steps = 4ms) — step tracking
% GRU can predict ahead; classical estimators output delayed estimates
P6 = P;
P6.sensor_delay = 2;
P6.T = 10;
scenarios(6).name = 'Delay Step';
scenarios(6).cmd  = @(t) [deg2rad(25)*(t>1 & t<3); 0; 0];
scenarios(6).P    = P6;

% Scenario 7: Sensor delay (3 steps = 6ms) + high-gain fast tracking
% Delay + high bandwidth → estimation lag is catastrophic for classical
P7 = P;
P7.sensor_delay = 3;
P7.K_rate = diag([35, 28, 18]);
P7.T = 10;
scenarios(7).name = 'Delay+HiGain';
scenarios(7).cmd  = @(t) [deg2rad(20)*sin(2*pi*2*t); ...
                           deg2rad(15)*sin(2*pi*3*t); ...
                           deg2rad(8)*sin(2*pi*1.5*t)];
scenarios(7).P    = P7;

% Scenario 8: Delay + model mismatch
P8 = P;
P8.sensor_delay = 2;
P8.B0_ctrl = P.B0 * 0.6;          % 40% mismatch
P8.K_rate  = diag([25, 20, 12]);
P8.T = 10;
scenarios(8).name = 'Delay+Mismatch';
scenarios(8).cmd  = @(t) [deg2rad(25)*(t>1 & t<4); ...
                           deg2rad(15)*(t>2 & t<5); ...
                           0];
scenarios(8).P    = P8;

% Scenario 9: Delay + accel disturbance + higher noise
P9 = P;
P9.sensor_delay = 2;
P9.dist_accel_std = [3; 2; 2];
P9.gyro_noise_std = deg2rad([0.2; 0.2; 0.15]);  % 2x noise
P9.K_rate  = diag([25, 20, 12]);
P9.T = 10;
scenarios(9).name = 'Delay+Dist+Noise';
scenarios(9).cmd  = @(t) [deg2rad(15)*sin(2*pi*1*t); ...
                           deg2rad(10)*sin(2*pi*1.5*t); ...
                           deg2rad(5)*sin(2*pi*0.5*t)];
scenarios(9).P    = P9;

% Scenario 10: Everything — delay + mismatch + disturbance + high gain + ramp
P10 = P;
P10.sensor_delay = 3;
P10.B0_ctrl_schedule.start  = P.B0;
P10.B0_ctrl_schedule.finish = P.B0 * 0.5;  % drifts to 50% error
P10.dist_accel_std = [2; 1.5; 1];
P10.gyro_noise_std = deg2rad([0.15; 0.15; 0.12]);
P10.K_rate = diag([30, 25, 15]);
P10.T = 15;
scenarios(10).name = 'Full Stress';
scenarios(10).cmd  = @(t) [deg2rad(20)*sin(2*pi*1*t); ...
                            deg2rad(12)*cos(2*pi*1.5*t); ...
                            deg2rad(6)*sin(2*pi*0.8*t)];
scenarios(10).P    = P10;

numScen = length(scenarios);
numEst  = length(estimators);

% ---- Run all combinations ----
fprintf('=== Closed-Loop Evaluation ===\n\n');
fprintf('%-16s | %-10s | %-22s | %-22s | %-22s\n', ...
    'Scenario', 'Estimator', 'Track RMSE (p,q,r) d/s', 'Peak Err (p,q,r) d/s', 'Act Rate RMS');
fprintf('%s\n', repmat('-', 1, 105));

all_logs = cell(numScen, numEst);

for s = 1:numScen
    for j = 1:numEst
        rng(42);  % same noise realization
        L = run_episode(scenarios(s).P, estimators{j}, scenarios(s).cmd);
        M = compute_closedloop_metrics(L, scenarios(s).P);
        all_logs{s, j} = L;

        fprintf('%-16s | %-10s | %5.2f %5.2f %5.2f       | %5.2f %5.2f %5.2f       | %5.2f %5.2f %5.2f\n', ...
            scenarios(s).name, est_names{j}, ...
            rad2deg(M.tracking_rmse'), rad2deg(M.peak_error'), rad2deg(M.actuator_rms'));
    end
    fprintf('%s\n', repmat('-', 1, 105));
end

% ---- Plot representative scenarios ----
colors = lines(numEst);

% Plot 1: Baseline — Roll Step (Scenario 1)
figure('Name', 'CL Baseline Roll Step', 'Position', [50 50 1000 500]);
for j = 1:numEst
    Lj = all_logs{1, j};
    plot(Lj.t, rad2deg(Lj.omega_meas(1,:)), 'Color', colors(j,:), ...
         'LineWidth', 1.0, 'DisplayName', est_names{j}); hold on;
end
plot(all_logs{1,1}.t, rad2deg(all_logs{1,1}.omega_cmd(1,:)), 'k--', ...
     'LineWidth', 1.5, 'DisplayName', 'Command');
xlabel('Time (s)'); ylabel('Roll Rate (deg/s)');
title('Baseline: Step Tracking (Roll, no delay)');
legend('Location', 'best'); grid on;

% Plot 2: Delay + High Gain (Scenario 7) — all 3 axes
if numScen >= 7
    axLabels = {'Roll', 'Pitch', 'Yaw'};
    figure('Name', 'CL Delay+HiGain', 'Position', [50 100 1200 700]);
    for ax = 1:3
        subplot(3, 1, ax);
        for j = 1:numEst
            Lj = all_logs{7, j};
            plot(Lj.t, rad2deg(Lj.omega_meas(ax,:)), 'Color', colors(j,:), ...
                 'LineWidth', 1.0, 'DisplayName', est_names{j}); hold on;
        end
        plot(all_logs{7,1}.t, rad2deg(all_logs{7,1}.omega_cmd(ax,:)), 'k--', ...
             'LineWidth', 1.5, 'DisplayName', 'Command');
        ylabel([axLabels{ax} ' (deg/s)']);
        if ax == 1, title('Delay + High Gain: Tracking (3-step delay, aggressive gains)'); end
        if ax == 3, xlabel('Time (s)'); end
        legend('Location', 'best'); grid on;
    end
end

% Plot 3: Full Stress (Scenario 10) — roll axis
if numScen >= 10
    figure('Name', 'CL Full Stress', 'Position', [50 150 1000 500]);
    for j = 1:numEst
        Lj = all_logs{10, j};
        plot(Lj.t, rad2deg(Lj.omega_meas(1,:)), 'Color', colors(j,:), ...
             'LineWidth', 1.0, 'DisplayName', est_names{j}); hold on;
    end
    plot(all_logs{10,1}.t, rad2deg(all_logs{10,1}.omega_cmd(1,:)), 'k--', ...
         'LineWidth', 1.5, 'DisplayName', 'Command');
    xlabel('Time (s)'); ylabel('Roll Rate (deg/s)');
    title('Full Stress: Roll (delay + mismatch + disturbance + ramp)');
    legend('Location', 'best'); grid on;
end

fprintf('\nClosed-loop evaluation complete.\n');
