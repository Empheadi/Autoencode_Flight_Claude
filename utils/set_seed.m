function set_seed(seed)
% SET_SEED  Set the random number generator seed for reproducibility.
%   set_seed(seed)
%
%   Input:
%     seed — integer seed value
    rng(seed);
end
