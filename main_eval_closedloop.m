% MAIN_EVAL_CLOSEDLOOP  Evaluate estimators in closed-loop INDI control.
%
%   Runs five scenarios with each estimator and compares tracking performance.

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

% Scenario 4: Disturbance rejection (hold zero, step dist at t=2)
P4 = P;
P4.dist_type = 'step';
P4.dist_std  = [10; 10; 5];
P4.T = 10;
scenarios(4).name = 'Dist Rejection';
scenarios(4).cmd  = @(t) [0; 0; 0];
scenarios(4).P    = P4;

% Scenario 5: Control effectiveness mismatch
P5 = P;
P5.B0_ctrl = P.B0 * 0.8;   % controller underestimates by 20%
scenarios(5).name = 'B Mismatch';
scenarios(5).cmd  = @(t) [deg2rad(25)*(t>1 & t<3); 0; 0];
scenarios(5).P    = P5;

numScen = length(scenarios);
numEst  = length(estimators);

% ---- Run all combinations ----
fprintf('=== Closed-Loop Evaluation ===\n\n');
fprintf('%-15s | %-10s | %-22s | %-22s | %-22s\n', ...
    'Scenario', 'Estimator', 'Track RMSE (p,q,r) d/s', 'Peak Err (p,q,r) d/s', 'Act Rate RMS');
fprintf('%s\n', repmat('-', 1, 100));

all_logs = cell(numScen, numEst);

for s = 1:numScen
    for j = 1:numEst
        rng(42);  % same noise realization
        L = run_episode(scenarios(s).P, estimators{j}, scenarios(s).cmd);
        M = compute_closedloop_metrics(L, scenarios(s).P);
        all_logs{s, j} = L;

        fprintf('%-15s | %-10s | %5.2f %5.2f %5.2f       | %5.2f %5.2f %5.2f       | %5.2f %5.2f %5.2f\n', ...
            scenarios(s).name, est_names{j}, ...
            rad2deg(M.tracking_rmse'), rad2deg(M.peak_error'), rad2deg(M.actuator_rms'));
    end
    fprintf('%s\n', repmat('-', 1, 100));
end

% ---- Plot representative scenario (Scenario 1: Roll Step) ----
figure('Name', 'CL Step Tracking', 'Position', [50 50 1000 500]);
colors = lines(numEst);
for j = 1:numEst
    Lj = all_logs{1, j};
    plot(Lj.t, rad2deg(Lj.omega_meas(1,:)), 'Color', colors(j,:), ...
         'LineWidth', 1.0, 'DisplayName', est_names{j}); hold on;
end
plot(all_logs{1,1}.t, rad2deg(all_logs{1,1}.omega_cmd(1,:)), 'k--', ...
     'LineWidth', 1.5, 'DisplayName', 'Command');
xlabel('Time (s)'); ylabel('Roll Rate (deg/s)');
title('Closed-Loop Step Tracking (Roll)');
legend('Location', 'best');
grid on;

fprintf('\nClosed-loop evaluation complete.\n');
