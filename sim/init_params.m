function P = init_params()
% INIT_PARAMS  Return default simulation parameter struct.
%   P = init_params()
%
%   Output:
%     P — struct with all simulation, vehicle, actuator, sensor,
%         disturbance, controller, and estimator parameters.

    % ---- Vehicle inertia (kg·m^2) ----
    P.I  = diag([4000, 6000, 8000]);        % 3×3 diagonal inertia matrix
    P.dt = 0.002;                            % time step (s)  — 500 Hz
    P.T  = 30;                               % episode length (s)

    % ---- Aerodynamic moment model ----
    %   M_aero = A(V)*omega + B(V)*delta
    %   A(V) = A0 + AV*(V - V0),  B(V) = B0 + BV*(V - V0)
    P.V0 = 80;                               % nominal airspeed (m/s)

    P.A0 = [-4.0   0     0.5;               % damping + cross-coupling (1/s)
              0    -3.0   0  ;
              0.3   0    -1.5];
    P.AV = P.A0 * 0.005;                    % speed scheduling slope

    P.B0 = [ 60    0     5;                  % control effectiveness (rad/s^2/rad)
              0   -40     0;
              3     0   -20];
    P.BV = P.B0 * 0.003;                    % speed scheduling slope

    % ---- Nonlinear damping ----
    %   M_nl = -mu * ||omega||^2 * omega   (cubic damping)
    P.mu_nl = 0.05;

    % ---- Actuator model (first-order lag + limits) ----
    P.tau_act  = [0.03; 0.03; 0.04];        % time constants (s)           3×1
    P.delta_max = deg2rad([25; 20; 25]);     % position limits (rad)        3×1
    P.rate_max  = deg2rad([80; 60; 80]);     % rate limits (rad/s)          3×1

    % ---- Sensor model ----
    % Realistic UAV-grade MEMS gyro noise levels.
    % Previous values (0.5, 0.5, 0.3 deg/s) were too large — noise
    % amplification in backward-difference made INDI unusable.
    P.gyro_noise_std = deg2rad([0.1; 0.1; 0.08]);   % white noise std (rad/s)  3×1
    P.gyro_bias_std  = deg2rad([0.01; 0.01; 0.005]); % per-episode bias std    3×1
    P.gyro_quantize  = deg2rad(0.005);               % quantization step (rad/s)

    % ---- Disturbance torque ----
    P.dist_type      = 'band_limited';       % 'none','step','band_limited','sinusoidal','ramp'
    P.dist_std       = [5; 5; 3];            % Nm, per-axis std             3×1
    P.dist_bw        = 10;                   % bandwidth (Hz)
    P.dist_accel_std = [0; 0; 0];            % acceleration-level disturbance (rad/s^2), bypasses I

    % ---- INDI rate-loop controller ----
    P.K_rate   = diag([15, 12, 8]);          % proportional rate gains      3×3
    P.B0_ctrl  = P.B0;                       % controller's copy of B0      3×3

    % ---- Estimator tuning ----
    P.est_lpf_alpha = 0.95;                  % LPF coefficient for backdiff
    P.est_eso_bw    = 50;                    % ESO bandwidth (rad/s)
    P.est_cf_alpha  = 0.7;                   % complementary filter blend

    % ---- Learned estimator ----
    P.Nwin = 25;                             % window length for learned estimators
end
