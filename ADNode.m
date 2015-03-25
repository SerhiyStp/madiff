classdef ADNode < handle
    properties
        value
        grad
        tape
        func
    end

    methods
        function n = ADNode(tape, x, func)
            if nargin > 2; 
                n.func = func; 
                tape.add(n);
            end
            n.tape = tape;
            n.value = x;
        end

        function add(x, grad)
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
        
        function backprop(n)
            n.func(n);
        end
        
        function y = tanh(x)
            y = ADNode(x.tape, tanh(x.value), @(y) x.add(y.grad .* sech(x.value) .^ 2));
        end
        
        function y = sum(x)
            y = ADNode(x.tape, sum(x.value), @(y) x.add(y.grad));
        end
        
    end

end