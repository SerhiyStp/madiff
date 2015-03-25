classdef ADNode < handle
%% Node in the function evalution graph
    
    properties
        value % function value at this node
        grad % gradient accumulator
        func % callback function to update gradient of the parent nodes
        root % input node that holds the tape
        tape % sequence of evaluation steps
    end

    methods
        function y = ADNode(x, root, func)
        %% create new node
            if nargin > 1; 
                y.func = func; 
                y.root = root;
                root.tape{end+1} = y;
            else
                y.root = y;
                y.tape = {};
            end
            y.value = x;
        end

        function backprop(x, dy)
        %% backpropagate the gradient by evaluating the tape backwards
            if nargin > 1
                x.grad = dy;
            else
                x.grad = 1;
            end
            for k = length(x.root.tape):-1:1
                x.root.tape{k}.func(x.root.tape{k});
                x.root.tape(k) = [];
            end
        end
        
        function y = tanh(x)
            y = ADNode(tanh(x.value), x.root, @(y) x.add(y.grad .* sech(x.value) .^ 2));
        end
        
        function y = sum(x)
            y = ADNode(sum(x.value), x.root, @(y) x.add(y.grad));
        end
        
        function y = abs(x)
            y = ADNode(abs(x.value), x.root, @(y) x.add(y.grad .* sign(x.value)));
        end
        
    end
    
    methods (Access = private)
        function add(x, grad)
        %% accumulate the gradient, take sum of dimensions if needed
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            if size(x.grad, 1) == 1 && size(x.grad, 2) == 1
                x.grad = x.grad + sum(sum(grad));
            elseif size(x.grad, 1) == 1
                x.grad = x.grad + sum(grad, 1);
            elseif size(x.grad, 2) == 1
                x.grad = x.grad + sum(grad, 2);
            else
                x.grad = x.grad + grad;
            end
        end
    end

end