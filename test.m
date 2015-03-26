ad = ?ADNode;
for k = 1:length(ad.MethodList)
    m = ad.MethodList(k);
    if length(m.InputNames) == 1 && strcmp(m.InputNames{1}, 'x')
        fn = [m.Name '(x);'];
        f = @(x) eval(fn);
        f_grad = @(x) adiff(f, x);
        if checkgrad(f_grad, randn, 1e-6) < 1e-7
            fprintf('%s = passed\n', m.Name);
        else
            fprintf('%s = FAILED\n', m.Name);
        end
    elseif length(m.InputNames) == 2 && strcmp(m.InputNames{1}, 'x1')
        fn = [m.Name '(x(1), x(2));'];
        f = @(x) eval(fn);
        f_grad = @(x) adiff(f, x);

        s = num2str(randn);
        fn2 = [m.Name '(' s ', x);'];
        f2 = @(x) eval(fn2);
        f2_grad = @(x) adiff(f2, x);

        fn3 = [m.Name '(x, ' s ');'];
        f3 = @(x) eval(fn3);
        f3_grad = @(x) adiff(f3, x);

        if checkgrad(f_grad, randn(2,1), 1e-6) < 1e-7 && ...
                checkgrad(f2_grad, randn, 1e-6) < 1e-7 && ...
                checkgrad(f3_grad, randn, 1e-6) < 1e-7
            fprintf('%s = passed\n', m.Name);
        else
            fprintf('%s = FAILED\n', m.Name);
        end
    end
end