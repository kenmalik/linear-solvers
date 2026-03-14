function [x, iters] = cg(A, b, x, L, tol, max_iters, real_residual)
    b_norm = norm(b);

    r = b - A * x;
    r_norm = norm(r);

    d = L' \ (L \ r);
    delta_new = r' * d;

    iters = 0;
    while (iters < max_iters) && (r_norm / b_norm > tol)
        iters = iters + 1;

        q = A * d;
        alpha = delta_new / (d' * q);

        x = x + alpha * d;

        if real_residual == true
            r = b - A * x;
        else
            r = r - alpha * q;
        end

        r_norm = norm(r);
        fprintf("%.15e\n", r_norm);

        s = L' \ (L \ r);

        delta_old = delta_new;
        delta_new = r' * s;

        beta = delta_new / delta_old;

        d = s + beta * d;
    end
end
