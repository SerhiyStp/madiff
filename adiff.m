function [y dy] = adiff(f, x, dy)
%% calculate the gradient 
    if nargout == 1
        y = f(x);
    else
        if nargin < 3; 
            dy = 1; 
        end
        x = ADNode(x);
        y = f(x);
        dy = y.backprop(dy);
        y = y.value;
    end