# MADiff: Matlab Automatic Differentiation (reverse mode, OO)

For a matlab function like: 

    % banana function
    f = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);

Create a new function that calculates its gradient as well:

    % banana function with a gradient
    f_grad = @(x) adiff(f, x);

Calculated in reverse mode and takes only ~5x function time:

    % calculate the function value only
    tic; y = f_grad(rand(1e7, 1)); toc
    Elapsed time is 0.429374 seconds.
    
    % calculate the gradient too
    tic; [y dy] = f_grad(rand(1e7, 1)); toc
    Elapsed time is 1.967412 seconds.

Compare to the numerical approximation:

    % should be less than 1e-8
    checkgrad(f_grad, randn(1e3,1), 1e-6)
    7.2e-09

Can be plugged into minimum search directly:

    tic;  minimize(f_grad, randn(1e2,1), 6e2); toc
    Linesearch    600;  Value 0.000314
    Elapsed time is 10.891118 seconds.

