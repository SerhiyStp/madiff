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
        
        function y = sum(x, dim, flag)
            switch nargin
              case 3
                y = ADNode(sum(x.value, dim, flag), x.root, @(y) x.add(y.grad));
              case 2
                y = ADNode(sum(x.value, dim), x.root, @(y) x.add(y.grad));
              otherwise
                y = ADNode(sum(x.value), x.root, @(y) x.add(y.grad));
            end
        end
        
        function y = abs(x)
            y = ADNode(abs(x.value), x.root, @(y) x.add(y.grad .* sign(x.value)));
        end
        
        function y = acos(x)
            y = ADNode(acos(x.value), x.root, @(y) x.add(-y.grad ./ sqrt(1-x.value.^2)));
        end

        function y = asin(x)
            y = ADNode(asin(x.value), x.root, @(y) x.add(y.grad ./ sqrt(1-x.value.^2)));
        end

        function y = atan(x)
            y = ADNode(atan(x.value), x.root, @(y) x.add(y.grad ./ (1+x.value.^2)));
        end

        function y = cos(x)
            y = ADNode(cos(x.value), x.root, @(y) x.add(-y.grad .* sin(x.value)));
        end
        
        function y = exp(x)
            y = ADNode(exp(x.value), x.root, @(y) x.add(y.grad .* exp(x.value)));
        end
        
        function y = log(x)
            y = ADNode(log(x.value), x.root, @(y) x.add(y.grad ./ x.value));
        end
        
        function y = sin(x)
            y = ADNode(sin(x.value), x.root, @(y) x.add(y.grad .* cos(x.value)));
        end

        function y = sqrt(x)
            y = ADNode(sqrt(x.value), x.root, @(y) x.add(y.grad ./ sqrt(x.value) / 2));
        end
        
        function y = tan(x)
            y = ADNode(tan(x.value), x.root, @(y) x.add(y.grad .* sec(x.value) .^ 2));
        end

        function y = uminus(x)
            y = ADNode(-x.value, x.root, @(y) x.add(-y.grad));
        end

        function y = uplus(x)
            y = ADNode(x.value, x.root, @(y) x.add(y.grad));
        end
        
        function [varargout] = subsref(x, s)
            switch s(1).type
              case '()'
                varargout{1} = ADNode(x.value(s.subs{:}), x.root, @(y) x.subs_add(s.subs, y));
              otherwise
                [varargout{1:nargout}] = builtin('subsref', x, s);
            end
        end
        
        function y = subsasgn(x, s, varargin)
            switch s(1).type
              case '()'
                x.value(s.subs{:}) = varargin{1}.value;
                t = ADNode(x.value(s.subs{:}), x.root, @(y) varargin{1}.subs_move(s.subs, x));
                y = x;
              otherwise
                y = builtin('subsagn', x, s, varargin);
            end
        end
        
        function y = plus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@plus, x1.value, x2.value), x1.root, @(y) y.plus_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@plus, x1.value, x2), x1.root, @(y) x1.add(y.grad));
                end
            else
                y = ADNode(bsxfun(@plus, x1, x2.value), x2.root, @(y) x2.add(y.grad));
            end
        end

        function y = minus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@minus, x1.value, x2.value), x1.root, @(y) y.minus_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@minus, x1.value, x2), x1.root, @(y) x1.add(y.grad));
                end
            else
                y = ADNode(bsxfun(@minus, x1, x2.value), x2.root, @(y) x2.add(-y.grad));
            end
        end
        
        function y = mtimes(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value * x2.value, x1.root, @(y) y.mtimes_backprop(x1, x2));
                else
                    y = ADNode(x1.value * x2, x1.root, @(y) x1.add(bsxfun(@times, y.grad, x2)));
                end
            else
                y = ADNode(x1 * x2.value, x2.root, @(y) x2.add(bsxfun(@times, y.grad, x1)));
            end
        end

        function y = times(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@times, x1.value, x2.value), x1.root, @(y) y.times_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@times, x1.value, x2), x1.root, @(y) x1.add(bsxfun(@times, y.grad,x2)));
                end
            else
                y = ADNode(bsxfun(@times, x1, x2.value), x2.root, @(y) x2.add(bsxfun(@times, y.grad, x1)));
            end
        end
        
        function y = rdivide(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@rdivide, x1.value, x2.value), x1.root, @(y) y.rdivide_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@rdivide, x1.value, x2), x1.root, @(y) x1.add(bsxfun(@rdivide, y.grad, x2)));
                end
            else
                y = ADNode(bsxfun(@rdivide, x1, x2.value), x2.root, ...
                           @(y) x2.add(- y.grad .* bsxfun(@rdivide, x1, x2.value .^ 2)));
            end
        end

        function y = mrdivide(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value / x2.value, x1.root, @(y) y.mrdivide_backprop(x1, x2));
                else
                    y = ADNode(x1.value / x2, x1.root, @(y) x1.add(y.grad / x2));
                end
            else
                y = ADNode(x1 / x2.value, x2.root, @(y) x2.add(- y.grad .* x1 / x2.value .^ 2));
            end
        end
        
        function y = mpower(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value ^ x2.value, x1.root, @(y) y.mpower_backprop(x1, x2));
                else
                    y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad * x1.value ^ (x2-1) * x2));
                end
            else
                t = x1 ^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(y.grad * t * log(x1)));
            end
        end

        function y = power(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value .^ x2.value, x1.root, @(y) y.power_backprop(x1, x2));
                else
                    y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add(y.grad .* x1.value .^ (x2-1) .* x2));
                end
            else
                t = x1 .^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(y.grad .* t .* log(x1)));
            end
        end
        
        function y = length(adn)
            y = length(adn.value);
        end
        
        function y = size(adn, dim)
            if nargin < 2;
                y = size(adn.value);
            else
                y = size(adn.value, dim);
            end
        end
        
        function y = bsxfun(op, x1, x2)
            switch func2str(op)
              case 'minus'
                y = minus(x1, x2);
              case 'plus'
                y = plus(x1, x2);
              case 'times'
                y = times(x1, x2);
              case 'rdivide'
                y = rdivide(x1, x2);
              otherwise
                assert(false, 'not implemented');
            end
        end
        
        function y = min(x1, x2)
            if nargin < 2
                [m, k] = min(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = min(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = min(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = min(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end
        
        function y = max(x1, x2)
            if nargin < 2
                [m, k] = max(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = max(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = max(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = max(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end
        
% end
% eq
% ge
% gt
% le
% lt
% ne
% norm
% sort
% vertcat
% horzcat
        
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
        
        function subs_add(x, subs, y)
        %% accumulate the gradient with subscripts
            grad = y.grad;
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            old = x.grad(subs{:});
            if size(old, 1) == 1 && size(old, 2) == 1
                x.grad(subs{:}) = old + sum(sum(grad));
            elseif size(old, 1) == 1
                x.grad(subs{:}) = old + sum(grad, 1);
            elseif size(old, 2) == 1
                x.grad(subs{:}) = old + sum(grad, 2);
            else
                x.grad(subs{:}) = old + grad;
            end
        end

        function subs_match(x, subs, y)
        %% accumulate the gradient with subscripts
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            if size(x.grad) == [1, 1]
                x.grad = x.grad + sum(y.grad(subs{:}));
            else
                x.grad(subs{:}) = x.grad(subs{:}) + y.grad(subs{:});
            end
        end

        function subs_move(x, subs, y)
        %% accumulate the gradient with subscripts
            grad = y.grad(subs{:});
            y.grad(subs{:}) = 0;
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            old = x.grad;
            if size(old, 1) == 1 && size(old, 2) == 1
                x.grad = old + sum(sum(grad));
            elseif size(old, 1) == 1
                x.grad = old + sum(grad, 1);
            elseif size(old, 2) == 1
                x.grad = old + sum(grad, 2);
            else
                x.grad = old + grad;
            end
        end
        
        function plus_backprop(y, x1, x2)
            x1.add(y.grad);
            x2.add(y.grad);
        end
        
        function minus_backprop(y, x1, x2)
            x1.add(y.grad);
            x2.add(-y.grad);
        end
    
        function mtimes_backprop(y, x1, x2)
            x1.add(y.grad * x2.value);
            x2.add(y.grad * x1.value);
        end
    
        function times_backprop(y, x1, x2)
            x1.add(bsxfun(@times, y.grad, x2.value));
            x2.add(bsxfun(@times, y.grad, x1.value));
        end
        
        function rdivide_backprop(y, x1, x2)
            x1.add(bsxfun(@rdivide, y.grad, x2.value));
            x2.add(-y.grad .* bsxfun(@rdivide, x1.value, x2.value .^ 2));
        end
    
        function mrdivide_backprop(y, x1, x2)
            x1.add(y.grad / x2.value);
            x2.add(-y.grad .* x1.value / x2.value .^ 2);
        end
        
        function mpower_backprop(y, x1, x2)
            x1.add(y.grad * x1.value ^ (x2.value-1) * x2.value);
            x2.add(y.grad * y.value * log(x1.value));
        end
    
        function power_backprop(y, x1, x2)
            x1.add(y.grad .* x1.value .^ (x2.value-1) .* x2.value);
            x2.add(y.grad .* y.value .* log(x1.value));
        end
        
        function minmax_backprop(y, x1, x2)
            x1.subs_match({find(y.value == x1.value)}, y);
            x2.subs_match({find(y.value == x2.value)}, y);
        end
    
    end

end