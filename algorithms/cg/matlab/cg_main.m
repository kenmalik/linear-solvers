function [] = cg_main(options)
    arguments
        options.dataset string = "1138_bus"
    end

    load("data/" + options.dataset + ".mat", "Problem");
    A = Problem.A;
    n = size(A, 1);

    load("data/" + options.dataset + "_ichol.mat", "L")

    b = ones(n, 1);
    x = zeros(n, 1);

    cg(A, b, x, L, 1e-6, n, true);
end
