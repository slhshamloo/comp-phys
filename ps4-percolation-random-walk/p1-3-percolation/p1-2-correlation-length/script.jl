"""script for running overnight to collect data"""

cd(dirname(@__FILE__))
include("../Percolation.jl")
using Plots, ColorSchemes, LaTeXStrings, DelimitedFiles, BenchmarkTools
figpath = "../../fig/percolation/"

for len in (10, 20, 40, 80, 160)
    samples, runs = 50, 1000

    probs = collect(range(0, 1, length=samples))
    gyration_avg, gyration_σ = zeros(samples), zeros(samples)

    for run in 1:runs
        randgrid = rand(len, len)
        for sample in 1:samples
            grid = randgrid .<= probs[sample]
            gyration = Percolation.clustgyration(grid)
            gyration_avg[sample] += gyration
            gyration_σ[sample] += gyration ^ 2
        end
    end

    gyration_avg ./= runs
    # absolute value added to ensure no negative numbers resulting from floating-point error at small σ
    gyration_σ = .√(abs.(gyration_σ ./ runs - gyration_avg .^ 2))

    # save data
    open("correlation-length-data/gyration-full-$len.txt", "w") do io
        writedlm(io, [probs gyration_avg gyration_σ])
    end

    p = scatter(probs, gyration_avg, yerr=gyration_σ, markerstrokecolor=:red,
    color=:mediumblue, lw=2, legend=false, fontfamily="Computer Modern",
    title=L"L=%$len,\ \textrm{averaged\ over\ %$runs \ runs}", xlabel=L"p", ylabel=L"\xi",
    titlefontsize=18, tickfontsize=10, labelfontsize=18)
    plot!(p, probs, gyration_avg, color=:mediumblue, lw=2)

    savefig(p, figpath * "gyration-full-$len.pdf")

    # use full range data to find narrow range for finding p_c and xi_max with better precision
    maxgyr = argmax(gyration_avg)
    gyrstep = probs[2] - probs[1]
    minprob, maxprob = probs[maxgyr - 1] - gyrstep / 2, probs[maxgyr + 1] + gyrstep / 2

    samples, runs = 50, 10000

    probs = collect(range(minprob, maxprob, length=samples))
    gyration_avg, gyration_σ = zeros(samples), zeros(samples)

    for run in 1:runs
        randgrid = rand(len, len)
        for sample in 1:samples
            grid = randgrid .<= probs[sample]
            gyration = Percolation.clustgyration(grid)
            gyration_avg[sample] += gyration
            gyration_σ[sample] += gyration ^ 2
        end
    end

    gyration_avg ./= runs
    # absolute value added to ensure no negative numbers resulting from floating-point error at small σ
    gyration_σ = .√(abs.(gyration_σ ./ runs - gyration_avg .^ 2))

    # save data
    open("correlation-length-data/gyration-zoom-$len.txt", "w") do io
        writedlm(io, [probs gyration_avg gyration_σ])
    end

    # find peak
    argpc = argmax(gyration_avg)
    pc = probs[argpc]
    ximax = gyration_avg[argpc]
    ximaxerr = gyration_σ[argpc]

    # save peak data
    open("correlation-length-data/gyration-peak.txt", "a") do io
        writedlm(io, [len pc ximax ximaxerr])
    end

    p = plot(probs, gyration_avg, ribbon=gyration_σ, fillcolor=:lightskyblue,
        color=:mediumblue, marker=:circle, lw=3, legend=false, fontfamily="Computer Modern",
        title=L"L=%$len,\ \textrm{averaged\ over\ %$runs \ runs}", xlabel=L"p", ylabel=L"\xi",
        titlefontsize=18, tickfontsize=10, labelfontsize=18)

    vline!(p, [pc], color=:red, linestyle=:dash)
    hline!(p, [ximax], color=:red, linestyle=:dash)

    annotetey = ylims(p)[1] + (ylims(p)[2] - ylims(p)[1]) / 15
    annotatex = pc + (xlims(p)[2] - xlims(p)[1]) / 100
    annotate!(p, annotatex, annotetey, text(L"p_c=%$(round(pc, digits=3))",
        :blueviolet, :left, :dash, 12))

    savefig(p, figpath * "gyration-zoom-$len.pdf")
end
