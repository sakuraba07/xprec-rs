using LegendrePolynomials
using FastGaussQuadrature
using Plots

function gauss_cheby_theta(::Type{T}, n::Integer) where T <: AbstractFloat
    θ = zeros(T, n)
    fact = T(π) / n
    for i = 0:n-1
        θ[i+1] = (n - i - 0.5) * fact
    end
    θ
end

function gauss_legendre_theta(θ0::AbstractVector{<:AbstractFloat}, nit::Integer=10)
    θ = copy(θ0)
    d1 = similar(θ)
    n = length(θ)
    for _ in 1:nit
        sc = sincos.(θ)
        s, c = first.(sc), last.(sc)
        p0 = Pl.(c, n-1)
        p1 = Pl.(c, n)
        d1 = @. n / s * (c * p1 - p0)
        θ -= @. p1 / d1
    end
    w = @. 2 / d1^2
    θ, w
end

θ0 = gauss_cheby_theta(Float64, 16);
θ, w = gauss_legendre_theta(θ0, 20);
x = cos.(θ)
xr, wr = gausslegendre(length(θ));

l = @layout [a b]

p1 = plot(@.(abs((x - xr) / xr) / eps(x) + 0.01), yscale=:log10, label="rel error (ulps)")
plot!(@.(abs(x - xr) / eps(x) + 0.01), yscale=:log10, label="abs error (multiple of ϵ)")
ylims!(1, 1000)
title!("Nodes")

p2 = plot(@.(abs((w - wr) / wr) / eps(x) + 0.01), yscale=:log10, label="rel error (ulps)")
plot!(@.(abs(w - wr) / eps(x) + 0.01), label="abs error (multiple of ϵ)")
ylims!(1, 1000)
title!("Weights")

p = plot(p1, p2; layout=l)
savefig(p, "gauss.pdf")
