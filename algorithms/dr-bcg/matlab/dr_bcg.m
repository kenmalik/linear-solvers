function [X, iters] = dr_bcg(A, B, X, L, tol, max_iters)
    R = B - A * X;
    [w,sigma] = qr(L^-1 * R,'econ');
    s = (L^-1)' * w;

    b1_norm = norm(B(:,1));

    iters = 0;
    for k = 1:max_iters
        iters = iters + 1;

        xi = (s' * A * s)^-1;
        X = X + s * xi * sigma;

        r_norm = norm(B(:,1) - A * X(:,1));
        fprintf("%.15e\n", r_norm / b1_norm);

        if r_norm / b1_norm < tol
            break
        else
            [w,zeta] = qr(w - (L^-1) * A * s * xi,'econ');
            s = (L^-1)' * w + s * zeta';
            sigma = zeta * sigma;
        end 
    end 
end
