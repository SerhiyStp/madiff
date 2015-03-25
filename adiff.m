function [y dy] = adiff(f, x)
    if nargout == 1
        y = f(x);
    else
        adt = ADTape(x);
        y = f(adt.x);
        y.grad = 1;
        adt.backprop;
        y = y.value;
        dy = adt.x.grad;
    end