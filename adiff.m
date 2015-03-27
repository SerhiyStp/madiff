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
        y.backprop(dy);
        y = y.value;
        dy = x.grad;
        if size(dy) ~= size(x)
            if size(dy) == [1, 1]
                dy = repmat(dy, size(x));
            elseif size(dy, 1) == 1 && size(x, 1) ~= 1
                dy = repmat(dy, size(x, 1), 1);
            elseif size(dy, 2) == 1 && size(x, 2) ~= 1
                dy = repmat(dy, 1, size(x, 2));
            end
        end
    end