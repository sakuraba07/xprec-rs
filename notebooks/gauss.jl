using DoubleFloats
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

# ============= Gauss new

function gauss_cheby_theta2(::Type{T}, n::Integer) where T <: AbstractFloat
    θ = zeros(T, n)
    θ′ = similar(θ)
    fact = T(π) / n
    for i = 0:n-1
        θ[i+1] = (n - i - 0.5) * fact
        θ′[i+1] = (n/2 - i - 0.5) * fact
    end
    θ, θ′
end

function gauss_legendre_theta2(θ₀::AbstractVector{<:AbstractFloat}, θ₀′::AbstractVector{<:AbstractFloat}, nit::Integer=10)
    θ = copy(θ₀)
    θ′ = copy(θ₀′)
    d1 = similar(θ)
    n = length(θ)
    for _ in 1:nit
        for i in 1:n
            if abs(θ[i]) < abs(θ′[i])
                sc = sincos(θ[i])
                s, c = sc
            else
                # θ′ = θ - π/2, thus:
                # cos(θ) = -sin(θ′); sin(θ) = cos(θ′)
                sc = sincos(θ′[i])
                c = -sc[1]
                s = sc[2]
            end
            p0 = Pl(c, n-1)
            p1 = Pl(c, n)
            d1[i] = n / s * (c * p1 - p0)
            θ[i] -= p1 / d1[i]
            θ′[i] -= p1 / d1[i]
        end
    end
    w = @. 2 / d1^2
    θ, θ′, w
end

n = 100
θ₀, θ₀′ = gauss_cheby_theta2(Float64, n)
@show maximum(abs, θ₀ - (θ₀′ .+ π/2))
θ, θ′, w = gauss_legendre_theta2(θ₀, θ₀′, 20)
x = @. ifelse(abs(θ) < abs(θ′), cos(θ), -sin(θ′))

#xr, wr = gausslegendre(length(θ))
θ0r = gauss_cheby_theta(Double64, n)
θr, wr = gauss_legendre_theta(θ0r, 20)
xr = cos.(θr);

θi, wi = gauss_legendre_theta(θ₀, 20)
xi = cos.(θi);

l = @layout [a b]

#p1 = plot(@.(abs((x - xr) / xr) / eps(x) + 0.01), yscale=:log10, label="rel error (ulps)")
p1 = plot(@.(abs(xi - xr) / eps(x) + 0.01), yscale=:log10, label="abs error (old method)")
plot!(@.(abs(x - xr) / eps(x) + 0.01), yscale=:log10, label="abs error (multiple of ϵ)")
ylims!(.1, 1000)
title!("Nodes")

#p2 = plot(@.(abs((w - wr) / wr) / eps(x) + 0.01), yscale=:log10, label="rel error (ulps)")
p2 = plot(@.(abs(wi - wr) / eps(x) + 0.01), yscale=:log10, label="abs error (old method)")
plot!(@.(abs(w - wr) / eps(x) + 0.01), label="abs error (multiple of ϵ)")
ylims!(.1, 1000)
title!("Weights")

p = plot(p1, p2; layout=l)
savefig(p, "gauss-new.pdf")
