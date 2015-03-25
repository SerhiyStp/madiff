% f = @(x) sum(100*(x(2:length(x))-x(1:length(x)-1).^2).^2 + (1-x(1:length(x)-1)).^2);

f = @(x) sum(tanh(x));
f_grad = @(x) adiff(f, x);
checkgrad(f_grad, rand(1e3, 1), 1e-6) < 1e-7

%f = @(x) sum(100*(x(2)-x(1).^2).^2 + (1-x(1)).^2);
%f_grad = @(x) adiff(f, x);
%checkgrad(f_grad, rand(2, 1), 1e-6)

