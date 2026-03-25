function y = saturate(x, lo, hi)
% SATURATE  Clamp each element of x to [lo, hi].
%   y = saturate(x, lo, hi)
%
%   Inputs:
%     x  — numeric array (any size)
%     lo — lower bound (scalar or same size as x)
%     hi — upper bound (scalar or same size as x)
%
%   Output:
%     y  — clamped array, same size as x
    y = max(min(x, hi), lo);
end
