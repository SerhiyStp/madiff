% banana function
f = @(x) sum(100*(x(2:length(x))-x(1:length(x)-1).^2).^2 + (1-x(1:length(x)-1)).^2);

% banana function with a gradient
f_grad = @(x) adiff(f, x);

disp('calculate the function value only');
tic; y = f_grad(rand(1e7, 1)); toc
    
disp('calculate the gradient too');
tic; [y dy] = f_grad(rand(1e7, 1)); toc

% compare gradient to the numeric approximation 
fprintf('error norm = %.2g (should be less than 1e-8)\n', checkgrad(f_grad, randn(1e3,1), 1e-6))

disp('can be plugged into minimum search directly')
tic;  minimize(f_grad, randn(1e2,1), 6e2); toc

