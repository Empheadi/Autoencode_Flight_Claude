% MAIN_TRAIN_GRU  Build windows from training data and train the GRU residual network.

rng(42);
addpath('sim', 'estimators', 'utils', 'data', 'models');

P = init_params();
Nwin = P.Nwin;
base_est = @est_eso_step;

% ---- Load and build windows ----
fprintf('Building training windows...\n');
X_all = [];  Y_res_all = [];

train_files = dir('data/train/run_*.mat');
for i = 1:length(train_files)
    d = load(fullfile(train_files(i).folder, train_files(i).name));
    P_run = d.L.params_run;
    [X_i, Y_res_i, ~, ~] = build_windows(d.L, P_run, base_est, Nwin);
    X_all     = cat(3, X_all, X_i);
    Y_res_all = cat(2, Y_res_all, Y_res_i);
    if mod(i, 50) == 0
        fprintf('  Loaded %d / %d train runs\n', i, length(train_files));
    end
end

fprintf('Building validation windows...\n');
X_val = [];  Y_res_val = [];

val_files = dir('data/val/run_*.mat');
for i = 1:length(val_files)
    d = load(fullfile(val_files(i).folder, val_files(i).name));
    P_run = d.L.params_run;
    [X_i, Y_res_i, ~, ~] = build_windows(d.L, P_run, base_est, Nwin);
    X_val     = cat(3, X_val, X_i);
    Y_res_val = cat(2, Y_res_val, Y_res_i);
end

% ---- Normalize ----
fprintf('Normalizing...\n');
[X_all, norm_stats] = normalize_data(X_all);
[X_val, ~]          = normalize_data(X_val, norm_stats);

% ---- Train ----
fprintf('Training GRU residual network...\n');
opts.maxEpochs     = 50;
opts.miniBatchSize = 128;
opts.initialLR     = 1e-3;

[net, info] = train_gru_residual(X_all, Y_res_all, X_val, Y_res_val, opts);

% ---- Save ----
save('models/gru_residual_trained.mat', 'net', 'norm_stats', 'info', '-v7.3');
fprintf('GRU residual model saved to models/gru_residual_trained.mat\n');

% ---- Plot training curve ----
figure('Name', 'GRU Training');
plot(1:opts.maxEpochs, info.trainLoss, 'b-', 'LineWidth', 1.2); hold on;
plot(1:opts.maxEpochs, info.valLoss, 'r--', 'LineWidth', 1.2);
xlabel('Epoch'); ylabel('MSE Loss');
legend('Train', 'Validation');
title('GRU Residual Training');
grid on;
