function d = checkgrad(f, X, e);

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad(f, X, e)
%
% where X is the argument and e is the small perturbation used for the finite
% differences. The function f should be of the type 
%
% [fX, dfX] = f(X)
%
% where fX is the function value and dfX is a vector of partial derivatives.
%
% Carl Edward Rasmussen, 2001-08-01.

[y dy] = f(X);                               % get the partial derivatives dy

dh = zeros(length(X),1) ;
for j = 1:length(X)
  dx = zeros(size(X));
  dx(j) = dx(j) + e;                             % perturb a single dimension
  y2 = f(X+dx);
  y1 = f(X-dx);
  dh(j) = (y2 - y1)/(2*e);
end

%disp([dy dh])                                         % print the two vectors
d = norm(dh-dy)/(norm(dh+dy)+eps);     % return norm of diff divided by norm of sum
