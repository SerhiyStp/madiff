# adiff
Automatic Differentiation for Matlab (OO, reverse mode)

For a matlab function like: 

  f = @(x) sum(tanh(x));

Create a function that will calculate a gradient as well:

  f_grad = @(x) adiff(f, x);

Calculated in reverse mode, so this should take less than a second:

  [y dy] = f_grad(rand(1e7, 1));