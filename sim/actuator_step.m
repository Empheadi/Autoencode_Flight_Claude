function delta_next = actuator_step(delta, delta_cmd, P)
% ACTUATOR_STEP  First-order actuator with rate and position limits.
%
%   delta_next = actuator_step(delta, delta_cmd, P)
%
%   Inputs:
%     delta     — current actuator positions (3×1, rad)
%     delta_cmd — commanded positions (3×1, rad)
%     P         — parameter struct (fields: tau_act, rate_max, delta_max, dt)
%
%   Output:
%     delta_next — actuator positions after one time step (3×1, rad)

    % First-order lag dynamics
    delta_dot = (delta_cmd - delta) ./ P.tau_act;   % 3×1

    % Rate limiting
    delta_dot = max(min(delta_dot, P.rate_max), -P.rate_max);

    % Integrate
    delta_next = delta + P.dt * delta_dot;

    % Position limiting
    delta_next = max(min(delta_next, P.delta_max), -P.delta_max);
end
