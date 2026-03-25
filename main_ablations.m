% MAIN_ABLATIONS  Run ablation studies comparing architecture variants.
%
%   A1: Direct prediction (no residual) vs residual
%   A2: Remove actuator inputs
%   A3: GRU residual (V1) vs autoencoder (V2)
%   A4: Window length sensitivity (Nwin = 15, 25, 40)
%
%   For each variant, trains a new model and evaluates offline + closed-loop.

rng(42);
addpath('sim', 'estimators', 'utils', 'data', 'models', 'eval');

P = init_params();
base_est = @est_eso_step;

% ---- Helper: train and evaluate a GRU variant ----
% Returns offline RMSE (3×1) and closed-loop tracking RMSE (3×1)

fprintf('=== Ablation Studies ===\n\n');
results_table = {};

% ---- Load test data once ----
test_files = dir('data/test/run_*.mat');
numTest = min(length(test_files), 20);  % use subset for speed

% ---- A1: Direct prediction vs residual ----
fprintf('--- A1: Direct prediction (no residual) ---\n');
% Train a GRU that directly predicts omega_dot_true instead of residual
[X_tr, Y_res_tr, Y_dir_tr, base_tr] = load_all_windows('data/train', P, base_est, P.Nwin);
[X_vl, Y_res_vl, Y_dir_vl, base_vl] = load_all_windows('data/val', P, base_est, P.Nwin);
[X_tr, ns] = normalize_data(X_tr);
[X_vl, ~]  = normalize_data(X_vl, ns);

% Direct: train on Y_direct
opts.maxEpochs = 30; opts.miniBatchSize = 128; opts.initialLR = 1e-3;
[net_direct, ~] = train_gru_residual(X_tr, Y_dir_tr, X_vl, Y_dir_vl, opts);
rmse_a1_direct = eval_net_offline(net_direct, ns, test_files, numTest, P, base_est, P.Nwin, 'direct');

% Residual (should already be trained, but retrain for fair comparison with same epochs)
[net_resid, ~] = train_gru_residual(X_tr, Y_res_tr, X_vl, Y_res_vl, opts);
rmse_a1_resid = eval_net_offline(net_resid, ns, test_files, numTest, P, base_est, P.Nwin, 'residual');

fprintf('A1 Direct RMSE  (deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a1_direct));
fprintf('A1 Residual RMSE(deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a1_resid));
results_table{end+1} = {'A1-Direct',  rmse_a1_direct};
results_table{end+1} = {'A1-Residual', rmse_a1_resid};

% ---- A2: Remove actuator inputs ----
fprintf('\n--- A2: Remove actuator inputs ---\n');
% Channels: omega_meas(3) + V(1) = 4 channels (drop delta_cmd, delta_true)
[X_tr_noact, Y_res_tr2, ~, ~] = load_all_windows('data/train', P, base_est, P.Nwin);
X_tr_noact = X_tr_noact([1:3, 10], :, :);  % keep omega(1:3) and V(10)
[X_vl_noact, Y_res_vl2, ~, ~] = load_all_windows('data/val', P, base_est, P.Nwin);
X_vl_noact = X_vl_noact([1:3, 10], :, :);

[X_tr_noact, ns2] = normalize_data(X_tr_noact);
[X_vl_noact, ~]   = normalize_data(X_vl_noact, ns2);

net_noact = build_gru_residual_net(4, P.Nwin);
[net_noact, ~] = train_gru_residual(X_tr_noact, Y_res_tr2, X_vl_noact, Y_res_vl2, opts);
rmse_a2 = eval_net_offline_custom(net_noact, ns2, [1:3, 10], test_files, numTest, P, base_est, P.Nwin);

fprintf('A2 No-actuator RMSE (deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a2));
fprintf('A2 Full-input  RMSE (deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a1_resid));
results_table{end+1} = {'A2-NoAct', rmse_a2};

% ---- A3: GRU residual vs autoencoder ----
fprintf('\n--- A3: GRU residual vs Autoencoder ---\n');
% Autoencoder
[X_tr3, ns3] = normalize_data(X_tr);  % reuse same norm (already normalized above)
% Actually X_tr is already normalized, so just use it
% Train AE
if exist('data/train/run_0001.mat', 'file')
    % Load omega_true for denoising target
    [~, ~, Y_dir_tr3, base_tr3] = load_all_windows('data/train', P, base_est, P.Nwin);
    [~, ~, Y_dir_vl3, base_vl3] = load_all_windows('data/val', P, base_est, P.Nwin);
    % Get omega_true at prediction index
    Y_omega_tr = load_omega_targets('data/train', P.Nwin);
    Y_omega_vl = load_omega_targets('data/val', P.Nwin);

    opts_ae = opts;
    opts_ae.lambda = [1.0, 0.3];
    [net_ae, ~] = train_autoencoder_observer(X_tr, Y_omega_tr, Y_res_tr, base_tr3, ...
        X_vl, Y_omega_vl, Y_res_vl, base_vl3, opts_ae);
    rmse_a3_ae = eval_ae_offline(net_ae, ns, test_files, numTest, P, base_est, P.Nwin);
    fprintf('A3 Autoencoder RMSE (deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a3_ae));
    fprintf('A3 GRU-Resid   RMSE (deg/s^2): [%.3f, %.3f, %.3f]\n', rad2deg(rmse_a1_resid));
    results_table{end+1} = {'A3-AE', rmse_a3_ae};
end

% ---- A4: Window length sensitivity ----
fprintf('\n--- A4: Window length sensitivity ---\n');
for Nw = [15, 25, 40]
    [Xw_tr, Yw_tr, ~, ~] = load_all_windows('data/train', P, base_est, Nw);
    [Xw_vl, Yw_vl, ~, ~] = load_all_windows('data/val', P, base_est, Nw);
    [Xw_tr, nsw] = normalize_data(Xw_tr);
    [Xw_vl, ~]   = normalize_data(Xw_vl, nsw);
    [net_w, ~] = train_gru_residual(Xw_tr, Yw_tr, Xw_vl, Yw_vl, opts);
    rmse_w = eval_net_offline(net_w, nsw, test_files, numTest, P, base_est, Nw, 'residual');
    fprintf('A4 Nwin=%2d  RMSE (deg/s^2): [%.3f, %.3f, %.3f]\n', Nw, rad2deg(rmse_w));
    results_table{end+1} = {sprintf('A4-Nwin%d', Nw), rmse_w};
end

% ---- Summary table ----
fprintf('\n=== Ablation Summary ===\n');
fprintf('%-15s | RMSE p (deg/s^2) | RMSE q | RMSE r\n', 'Variant');
fprintf('%s\n', repmat('-', 1, 55));
for i = 1:length(results_table)
    r = results_table{i};
    fprintf('%-15s | %8.4f         | %6.4f | %6.4f\n', r{1}, rad2deg(r{2}));
end

fprintf('\nAblation studies complete.\n');


%% ========== Local helper functions ==========

function [X, Y_res, Y_dir, base] = load_all_windows(folder, P, base_est, Nwin)
    files = dir(fullfile(folder, 'run_*.mat'));
    X = []; Y_res = []; Y_dir = []; base = [];
    for i = 1:length(files)
        d = load(fullfile(files(i).folder, files(i).name));
        P_run = d.L.params_run;
        [Xi, Yri, Ydi, bi] = build_windows(d.L, P_run, base_est, Nwin);
        X     = cat(3, X, Xi);
        Y_res = cat(2, Y_res, Yri);
        Y_dir = cat(2, Y_dir, Ydi);
        base  = cat(2, base, bi);
    end
end

function Y_omega = load_omega_targets(folder, Nwin)
    files = dir(fullfile(folder, 'run_*.mat'));
    Y_omega = [];
    for i = 1:length(files)
        d = load(fullfile(files(i).folder, files(i).name));
        N = size(d.L.omega_true, 2);
        Y_omega = cat(2, Y_omega, d.L.omega_true(:, Nwin:N));
    end
end

function rmse = eval_net_offline(net, ns, test_files, numTest, P, base_est, Nwin, mode)
    % mode: 'residual' or 'direct'
    rmse_all = zeros(3, numTest);
    for r = 1:numTest
        d = load(fullfile(test_files(r).folder, test_files(r).name));
        P_run = d.L.params_run;
        [X_i, ~, ~, base_i] = build_windows(d.L, P_run, base_est, Nwin);
        [X_i, ~] = normalize_data(X_i, ns);
        Y_pred = predict(net, dlarray(X_i, 'CTB'));
        Y_pred = extractdata(Y_pred);  % 3 × numSamples
        N = size(d.L.omega_true, 2);
        Y_true = d.L.omega_dot_true(:, Nwin:N);
        if strcmp(mode, 'residual')
            Y_pred = base_i + Y_pred;
        end
        rmse_all(:, r) = sqrt(mean((Y_true - Y_pred).^2, 2));
    end
    rmse = mean(rmse_all, 2);
end

function rmse = eval_net_offline_custom(net, ns, ch_idx, test_files, numTest, P, base_est, Nwin)
    rmse_all = zeros(3, numTest);
    for r = 1:numTest
        d = load(fullfile(test_files(r).folder, test_files(r).name));
        P_run = d.L.params_run;
        [X_i, ~, ~, base_i] = build_windows(d.L, P_run, base_est, Nwin);
        X_i = X_i(ch_idx, :, :);
        [X_i, ~] = normalize_data(X_i, ns);
        Y_pred = predict(net, dlarray(X_i, 'CTB'));
        Y_pred = extractdata(Y_pred);
        N = size(d.L.omega_true, 2);
        Y_true = d.L.omega_dot_true(:, Nwin:N);
        Y_pred = base_i + Y_pred;
        rmse_all(:, r) = sqrt(mean((Y_true - Y_pred).^2, 2));
    end
    rmse = mean(rmse_all, 2);
end

function rmse = eval_ae_offline(net, ns, test_files, numTest, P, base_est, Nwin)
    rmse_all = zeros(3, numTest);
    for r = 1:numTest
        d = load(fullfile(test_files(r).folder, test_files(r).name));
        P_run = d.L.params_run;
        [X_i, ~, ~, base_i] = build_windows(d.L, P_run, base_est, Nwin);
        [X_i, ~] = normalize_data(X_i, ns);
        [~, delta_od] = predict(net, dlarray(X_i, 'CTB'), 'Outputs', {'h1_out','h2_out'});
        delta_od = extractdata(delta_od);
        N = size(d.L.omega_true, 2);
        Y_true = d.L.omega_dot_true(:, Nwin:N);
        Y_pred = base_i + delta_od;
        rmse_all(:, r) = sqrt(mean((Y_true - Y_pred).^2, 2));
    end
    rmse = mean(rmse_all, 2);
end
