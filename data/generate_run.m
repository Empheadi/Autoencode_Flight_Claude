function L = generate_run(P, seed)
% GENERATE_RUN  Generate one simulation run with randomized conditions.
%
%   L = generate_run(P, seed)
%
%   Inputs:
%     P    — base parameter struct (from init_params)
%     seed — RNG seed for this run
%
%   Output:
%     L — log struct from run_episode, plus L.params_run (run-specific params)

    rng(seed);

    % ---- Randomize conditions ----
    P_run = P;
    P_run.gyro_noise_std = P.gyro_noise_std .* (1 + 0.3*randn);
    P_run.gyro_noise_std = max(P_run.gyro_noise_std, 0);  % ensure non-negative

    P_run.tau_act = P.tau_act .* (1 + 0.2*randn(3,1));
    P_run.tau_act = max(P_run.tau_act, 0.005);             % minimum 5 ms

    P_run.I = P.I .* (1 + 0.1*diag(randn(3,1)));

    P_run.V0 = P.V0 + 20*(2*rand - 1);                    % V ∈ [60, 100] m/s

    P_run.dist_std = P.dist_std .* (0.5 + rand);
    P_run.dist_std = max(P_run.dist_std, 0);

    % ---- Random excitation command ----
    omega_cmd_fun = make_random_excitation(P_run);

    % ---- Run with backdiff estimator ----
    L = run_episode(P_run, @est_backdiff_lpf_step, omega_cmd_fun);
    L.params_run = P_run;
end
