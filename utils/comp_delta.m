% Compute the difference of a vector.

function dx = comp_delta(x)
    dx = zeros(length(x) -1, 1);
    for i = 1:length(x) - 1
        dx(i) = x(i + 1) - x(i);
    end
end