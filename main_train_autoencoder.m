% MAIN_TRAIN_AUTOENCODER  Train the autoencoder observer (two-head) network.

rng(42);
addpath('sim', 'estimators', 'utils', 'data', 'models');

P = init_params();
Nwin   = P.Nwin;
stride = 10;           % subsample windows for speed
base_est = @est_eso_step;

% ---- Load and build windows (training) ----
fprintf('Building training windows (stride=%d)...\n', stride);
X_all = [];  Y_res_all = [];  Y_dir_all = [];  base_all = [];

train_files = dir('data/train/run_*.mat');
for i = 1:length(train_files)
    d = load(fullfile(train_files(i).folder, train_files(i).name));
    P_run = d.L.params_run;
    [X_i, Y_res_i, Y_dir_i, base_i] = build_windows(d.L, P_run, base_est, Nwin, stride);

    % omega_true at prediction indices for denoising head
    N = size(d.L.omega_true, 2);
    all_starts = 1 : stride : (N - Nwin + 1);
    pred_idx = all_starts + Nwin - 1;
    Y_omega_i = d.L.omega_true(:, pred_idx);

    X_all     = cat(3, X_all, X_i);
    Y_res_all = cat(2, Y_res_all, Y_res_i);
    Y_dir_all = cat(2, Y_dir_all, Y_omega_i);
    base_all  = cat(2, base_all, base_i);

    if mod(i, 50) == 0
        fprintf('  Loaded %d / %d train runs (%d samples)\n', ...
            i, length(train_files), size(X_all, 3));
    end
end
fprintf('Total training samples: %d\n', size(X_all, 3));

% ---- Validation windows ----
fprintf('Building validation windows...\n');
X_val = [];  Y_res_val = [];  Y_dir_val = [];  base_val = [];

val_files = dir('data/val/run_*.mat');
for i = 1:length(val_files)
    d = load(fullfile(val_files(i).folder, val_files(i).name));
    P_run = d.L.params_run;
    [X_i, Y_res_i, ~, base_i] = build_windows(d.L, P_run, base_est, Nwin, stride);

    N = size(d.L.omega_true, 2);
    all_starts = 1 : stride : (N - Nwin + 1);
    pred_idx = all_starts + Nwin - 1;
    Y_omega_i = d.L.omega_true(:, pred_idx);

    X_val     = cat(3, X_val, X_i);
    Y_res_val = cat(2, Y_res_val, Y_res_i);
    Y_dir_val = cat(2, Y_dir_val, Y_omega_i);
    base_val  = cat(2, base_val, base_i);
end
fprintf('Total validation samples: %d\n', size(X_val, 3));

% ---- Normalize ----
fprintf('Normalizing...\n');
[X_all, norm_stats] = normalize_data(X_all);
[X_val, ~]          = normalize_data(X_val, norm_stats);

% ---- Train ----
fprintf('Training autoencoder observer...\n');
opts.maxEpochs     = 20;
opts.miniBatchSize = 512;
opts.initialLR     = 1e-3;
opts.lambda        = [1.0, 0.3];

[net, info] = train_autoencoder_observer(X_all, Y_dir_all, Y_res_all, base_all, ...
    X_val, Y_dir_val, Y_res_val, base_val, opts);

% ---- Save ----
save('models/autoencoder_observer_trained.mat', 'net', 'norm_stats', 'info', '-v7.3');
fprintf('Autoencoder observer saved to models/autoencoder_observer_trained.mat\n');

% ---- Plot ----
figure('Name', 'AE Training');
plot(1:opts.maxEpochs, info.trainLoss, 'b-', 'LineWidth', 1.2); hold on;
plot(1:opts.maxEpochs, info.valLoss, 'r--', 'LineWidth', 1.2);
xlabel('Epoch'); ylabel('Loss');
legend('Train', 'Validation');
title('Autoencoder Observer Training');
grid on;
