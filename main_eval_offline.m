% MAIN_EVAL_OFFLINE  Evaluate all estimators on test data (offline).
%
%   Loads test runs, runs each estimator, computes metrics, and plots results.

rng(42);
addpath('sim', 'estimators', 'utils', 'data', 'eval', 'models');

P = init_params();

estimators = {@est_backdiff_lpf_step, @est_eso_step, @est_compfilt_step};
est_names  = {'BackDiff', 'ESO', 'CompFilt'};

% Add learned estimator if model exists
if exist('models/gru_residual_trained.mat', 'file')
    estimators{end+1} = @est_learned_step;
    est_names{end+1}  = 'GRU-Res';
end

numEst = length(estimators);

% ---- Load test data ----
test_files = dir('data/test/run_*.mat');
numTest = length(test_files);
fprintf('Found %d test runs.\n', numTest);

% Storage for aggregated metrics
all_metrics = cell(numTest, numEst);

for r = 1:numTest
    d = load(fullfile(test_files(r).folder, test_files(r).name));
    L = d.L;

    for j = 1:numEst
        % Run estimator on logged data
        N = size(L.omega_meas, 2);
        omega_dot_est = zeros(3, N);
        [~, es] = estimators{j}('init', [], [], P, []);
        for k = 1:N
            [omega_dot_est(:,k), es] = estimators{j}('step', ...
                L.omega_meas(:,k), L.delta_true(:,k), P, es);
        end

        all_metrics{r, j} = compute_offline_metrics(L.omega_dot_true, omega_dot_est, P.dt);
    end

    if mod(r, 20) == 0
        fprintf('  Processed %d / %d test runs\n', r, numTest);
    end
end

% ---- Aggregate metrics ----
fprintf('\n=== Offline Estimation Metrics (mean +/- std across test runs) ===\n');
fprintf('%-10s | %-22s | %-22s | %-22s | %-15s\n', ...
    'Estimator', 'RMSE (p,q,r) deg/s^2', 'MAE (p,q,r) deg/s^2', ...
    'PeakErr (p,q,r) deg/s^2', 'Delay (ms)');
fprintf('%s\n', repmat('-', 1, 95));

for j = 1:numEst
    rmse_all = zeros(numTest, 3);
    mae_all  = zeros(numTest, 3);
    peak_all = zeros(numTest, 3);
    delay_all = zeros(numTest, 3);
    for r = 1:numTest
        rmse_all(r,:)  = rad2deg(all_metrics{r,j}.rmse');
        mae_all(r,:)   = rad2deg(all_metrics{r,j}.mae');
        peak_all(r,:)  = rad2deg(all_metrics{r,j}.peak_err');
        delay_all(r,:) = all_metrics{r,j}.delay_sec' * 1000;
    end

    rmse_m = mean(rmse_all, 1);
    mae_m  = mean(mae_all, 1);
    peak_m = mean(peak_all, 1);
    delay_m = mean(delay_all, 1);

    fprintf('%-10s | %5.2f %5.2f %5.2f       | %5.2f %5.2f %5.2f       | %6.1f %6.1f %6.1f     | %4.1f %4.1f %4.1f\n', ...
        est_names{j}, rmse_m, mae_m, peak_m, delay_m);
end

% ---- Representative run plots ----
if numTest > 0
    d = load(fullfile(test_files(1).folder, test_files(1).name));
    L = d.L;
    N = size(L.omega_meas, 2);
    est_outputs = cell(1, numEst);
    met_cell    = cell(1, numEst);

    for j = 1:numEst
        omega_dot_est = zeros(3, N);
        [~, es] = estimators{j}('init', [], [], P, []);
        for k = 1:N
            [omega_dot_est(:,k), es] = estimators{j}('step', ...
                L.omega_meas(:,k), L.delta_true(:,k), P, es);
        end
        est_outputs{j} = omega_dot_est;
        met_cell{j} = compute_offline_metrics(L.omega_dot_true, omega_dot_est, P.dt);
    end

    % Time overlay
    plot_time_overlay(L, est_outputs, est_names, [2, 8]);

    % Frequency metrics
    plot_frequency_metrics(met_cell, est_names);

    % RMSE bar chart
    figure('Name', 'RMSE Bar');
    rmse_vals = zeros(3, numEst);
    for j = 1:numEst
        rmse_vals(:, j) = rad2deg(met_cell{j}.rmse);
    end
    bar(rmse_vals');
    set(gca, 'XTickLabel', est_names);
    legend('Roll', 'Pitch', 'Yaw');
    ylabel('RMSE (deg/s^2)');
    title('Estimation RMSE (representative run)');
    grid on;
end

fprintf('\nOffline evaluation complete.\n');
