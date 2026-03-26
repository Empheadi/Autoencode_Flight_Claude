function P_mc = randomize_params(P)
% RANDOMIZE_PARAMS  Apply random perturbations to parameters for Monte Carlo.
%
%   P_mc = randomize_params(P)
%
%   Input:
%     P — base parameter struct
%
%   Output:
%     P_mc — perturbed parameter struct

    P_mc = P;

    P_mc.gyro_noise_std = P.gyro_noise_std .* (1 + 0.5*randn);
    P_mc.gyro_noise_std = max(P_mc.gyro_noise_std, 0);

    P_mc.gyro_bias_std = P.gyro_bias_std .* (1 + 0.5*abs(randn));

    P_mc.tau_act = P.tau_act .* (1 + 0.3*randn(3,1));
    P_mc.tau_act = max(P_mc.tau_act, 0.005);

    P_mc.I = P.I .* (1 + 0.15*diag(randn(3,1)));

    P_mc.B0_ctrl = P.B0 .* (1 + 0.15*randn(size(P.B0)));

    P_mc.dist_std = P.dist_std .* (0.5 + 1.5*rand);
    P_mc.dist_std = max(P_mc.dist_std, 0);

    % Randomize sensor delay (0–3 steps)
    P_mc.sensor_delay = randi([0, 3]);

    % Randomize acceleration-level disturbance
    if rand > 0.5
        P_mc.dist_accel_std = [2; 1.5; 1] .* (0.5 + rand);
    else
        P_mc.dist_accel_std = [0; 0; 0];
    end
end
