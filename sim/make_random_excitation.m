function omega_cmd_fun = make_random_excitation(P)
% MAKE_RANDOM_EXCITATION  Generate a random piecewise rate-command profile.
%
%   omega_cmd_fun = make_random_excitation(P)
%
%   Builds a 3-axis command signal composed of doublets, steps, chirps,
%   and zero segments. Returns a function handle omega_cmd_fun(t) → 3×1.
%
%   Input:
%     P — parameter struct (fields: T, dt)
%
%   Output:
%     omega_cmd_fun — function handle: omega_cmd = f(t), returns 3×1 (rad/s)

    N = round(P.T / P.dt);
    signal = zeros(3, N);
    dt = P.dt;

    for ax = 1:3
        k = 1;
        while k <= N
            seg_type = randi(4);   % 1=doublet, 2=step, 3=chirp, 4=zero

            switch seg_type
                case 1  % Doublet
                    amp  = deg2rad(10 + 30*rand);           % [10, 40] deg/s
                    dur  = 0.2 + 0.8*rand;                  % [0.2, 1.0] s
                    npts = round(dur / dt);
                    half = round(npts / 2);
                    seg  = [amp*ones(1, half), -amp*ones(1, npts - half)];

                case 2  % Step
                    amp  = deg2rad(5 + 25*rand) * (2*(rand>0.5)-1);  % ±[5,30] deg/s
                    dur  = 0.5 + 1.5*rand;                  % [0.5, 2.0] s
                    npts = round(dur / dt);
                    seg  = amp * ones(1, npts);

                case 3  % Chirp
                    amp   = deg2rad(5 + 15*rand);            % [5, 20] deg/s
                    dur   = 3.0;                              % 3 s
                    npts  = round(dur / dt);
                    tvec  = (0:npts-1) * dt;
                    f0 = 0.5; f1 = 15;                       % Hz sweep range
                    seg = amp * sin(2*pi * (f0*tvec + (f1-f0)/(2*dur)*tvec.^2));

                case 4  % Zero (rest)
                    dur  = 0.3 + 1.0*rand;                   % [0.3, 1.3] s
                    npts = round(dur / dt);
                    seg  = zeros(1, npts);
            end

            % Append segment
            k_end = min(k + length(seg) - 1, N);
            signal(ax, k:k_end) = seg(1:(k_end - k + 1));
            k = k_end + 1;
        end
    end

    % Store precomputed signal; interpolate via nearest index
    t_vec = (0:N-1) * dt;
    omega_cmd_fun = @(t) interp_cmd(t, t_vec, signal);
end

function cmd = interp_cmd(t, t_vec, signal)
% Nearest-neighbour lookup of precomputed command signal.
    [~, idx] = min(abs(t_vec - t));
    cmd = signal(:, idx);   % 3×1
end
