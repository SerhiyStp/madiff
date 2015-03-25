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
    end
end