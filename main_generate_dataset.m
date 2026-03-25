% MAIN_GENERATE_DATASET  Generate training, validation, and test datasets.
%
%   Produces .mat files in data/train/, data/val/, data/test/.
%   Each file contains the log struct L from one simulation run with
%   randomized conditions.

addpath('sim', 'estimators', 'utils', 'data');

P = init_params();

% ---- Training: 300 runs, seeds 1–300 ----
fprintf('Generating training data...\n');
for i = 1:300
    L = generate_run(P, i);
    save(sprintf('data/train/run_%04d.mat', i), 'L', '-v7.3');
    if mod(i, 50) == 0
        fprintf('  Train: %d / 300\n', i);
    end
end

% ---- Validation: 50 runs, seeds 1001–1050 ----
fprintf('Generating validation data...\n');
for i = 1:50
    L = generate_run(P, 1000 + i);
    save(sprintf('data/val/run_%04d.mat', i), 'L', '-v7.3');
    if mod(i, 10) == 0
        fprintf('  Val: %d / 50\n', i);
    end
end

% ---- Test: 100 runs, seeds 2001–2100, wider uncertainty ----
fprintf('Generating test data...\n');
P_test = P;
% Widen ranges by 1.5x (generate_run applies its own randomization on top)
P_test.gyro_noise_std = P.gyro_noise_std * 1.5;
P_test.tau_act        = P.tau_act * 1.2;
for i = 1:100
    L = generate_run(P_test, 2000 + i);
    save(sprintf('data/test/run_%04d.mat', i), 'L', '-v7.3');
    if mod(i, 20) == 0
        fprintf('  Test: %d / 100\n', i);
    end
end

fprintf('Dataset generation complete.\n');
