% banana function
f = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);

% banana function with a gradient
f_grad = @(x) adiff(f, x);

fprintf('calculate the function value only: ');
tic; y = f_grad(rand(1e7, 1)); toc

x = randn(1e7, 1);
fprintf('\ncalculate the gradient too: ');
tic; [y dy] = f_grad(x); toc

fprintf('\ncompared to the handcrafted derivative calculation: ');
tic; [y1 dy1] = rosenbrock(x); toc

fprintf('vs. exact = %.2g \n', norm(dy-dy1)/(norm(dy+dy1)+eps)); 

% compare gradient to the numeric approximation 
fprintf('\nvs. numeric approx = %.2g (should be less than 1e-8)\n', checkgrad(f_grad, randn(1e3,1), 1e-6))

x0 = randn(1e2,1);
fprintf('\ncan be plugged into minimum search directly:\n')
tic;  minimize(f_grad, x0, 6e2); toc

fprintf('\nvs. exact:\n')
tic;  minimize(@rosenbrock, x0, 6e2); toc
