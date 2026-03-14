function [] = dr_bcg_main(options)
    arguments
        options.dataset string = "1138_bus"
        options.block_size (1, 1) int32 = 1
    end

    load("data/" + options.dataset + ".mat", "Problem");
    A = Problem.A;
    n = size(A, 1);

    load("data/" + options.dataset + "_ichol.mat", "L")

    b = ones(n, options.block_size);
    x = zeros(n, options.block_size);

    dr_bcg(A, b, x, L, 1e-6, n);
end
