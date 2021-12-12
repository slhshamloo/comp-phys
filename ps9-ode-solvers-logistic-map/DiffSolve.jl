module DiffSolve

export eulersolve, eulercromer, eulermid, verlet, velverlet, rk4

"""
Solve a system of first order ODEs dx/dt = f(t, x(t)) using the Euler method.
x(t) is a vector.

# Arguments
- `init`: initial x vector
- `func`: the function dx/dt = f(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
vector of all t and matrix containing x at all t (the (i, j) element in the matrix
represents the jth variable in the x vector x[j] at the ith time step t[i]).
"""
function eulersolve(func::Function, init::Vector{Float64}, start::Float64, stop::Float64;
        step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Matrix{Float64}(undef, length(ts), length(init))
    xs[1, :] = init

    x = init
    for (i, t) in enumerate(@view ts[1:end-1])
        x += step * func(t, x)
        xs[i + 1, :] = x
    end

    return ts, xs
end

"""
Solve a system of first order ODEs dx/dt = f(t, x(t)) using the 4th order Runge–Kutta
method (a.k.a. RK4, or Classic Runge–Kutta). x(t) is a vector.

# Arguments
- `init`: initial x vector
- `func`: the function dx/dt = f(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
vector of all t and matrix containing x at all t (the (i, j) element in the matrix
represents the jth variable in the x vector x[j] at the ith time step t[i]).
"""
function rk4(func::Function, init::Vector{Float64}, start::Float64, stop::Float64;
        step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Matrix{Float64}(undef, length(ts), length(init))
    xs[1, :] = init

    x = init
    for (i, t) in enumerate(@view ts[1:end-1])
        k1 = func(t, x)
        k2 = func(t + step / 2, x .+ step / 2 * k1)
        k3 = func(t + step / 2, x .+ step / 2 * k2)
        k4 = func(t + step, x .+ step * k3)

        x += step / 6 * (k1 + 2k2 + 2k3 + k4)
        xs[i + 1, :] = x
    end

    return ts, xs
end

"""
Solve a second order ODE d²x/dt² = a(t, x(t)) using the Euler–Cromer method
(a.k.a. the semi-implicit euler method, or the Newton–Størmer–Verlet).

# Arguments
- `initx`, `initv`: initial x (displacement) and dx/dt (velocity v)
- `acceleration`: the function d²x/dt² = a(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
t, x, and v (dx/dt) vectors.
"""
function eulercromer(acceleration::Function, initx::Float64, initv::Float64,
        start::Float64, stop::Float64; step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Vector{Float64}(undef, length(ts))
    vs = Vector{Float64}(undef, length(ts))
    xs[1] = initx
    vs[1] = initv

    x = initx
    v = initv
    for (i, t) in enumerate(@view ts[1:end-1])
        v += step * acceleration(t, x)
        x += step * v

        xs[i + 1] = x
        vs[i + 1] = v
    end

    return ts, xs, vs
end

"""
Solve a second order ODE d²x/dt² = a(t, x(t)) using the Midpoint method
(a modified euler method).

# Arguments
- `initx`, `initv`: initial x (displacement) and dx/dt (velocity v)
- `acceleration`: the function d²x/dt² = a(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
t, x, and v (dx/dt) vectors. Due to the way this method works, the v vector is the
velocity at midpoints between the time steps.
"""
function eulermid(acceleration::Function, initx::Float64, initv::Float64,
        start::Float64, stop::Float64; step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Vector{Float64}(undef, length(ts))
    vs = Vector{Float64}(undef, length(ts))
    xs[1] = initx
    vs[1] = initv - acceleration(start, initx) * step / 2

    x = initx
    v = vs[1]
    for (i, t) in enumerate(@view ts[1:end-1])
        v += step * acceleration(t, x)
        x += step * v

        xs[i + 1] = x
        vs[i + 1] = v
    end

    return ts, xs, vs
end

"""
Solve a second order ODE d²x/dt² = a(t, x(t)) using Verlet integration
(a.k.a. the Störmer–Verlet method).

# Arguments
- `initx`, `initv`: initial x (displacement) and dx/dt (velocity v)
- `acceleration`: the function d²x/dt² = a(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
t, x, and v (dx/dt) vectors.
"""
function verlet(acceleration::Function, initx::Float64, initv::Float64,
        start::Float64, stop::Float64; step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Vector{Float64}(undef, length(ts))
    vs = Vector{Float64}(undef, length(ts))

    xs[1] = initx
    vs[1] = initv
    xs[2] = initx + initv * step + acceleration(start, initx) * step * step / 2

    for i in 2:length(xs)-1
        xs[i + 1] = 2 * xs[i] - xs[i - 1] + acceleration(ts[i], xs[i]) * step * step
        vs[i + 1] = (xs[i + 1] - xs[i]) / step
    end

    return ts, xs, vs
end

"""
Solve a second order ODE d²x/dt² = a(t, x(t)) using the velocity Verlet method

# Arguments
- `initx`, `initv`: initial x (displacement) and dx/dt (velocity v)
- `acceleration`: the function d²x/dt² = a(t, x(t))
- `start`: initial t
- `stop`: final t

# Returns
t, x, and v (dx/dt) vectors.
"""
function velverlet(acceleration::Function, initx::Float64, initv::Float64,
        start::Float64, stop::Float64; step::Float64 = 0.01)
    ts = collect(range(start, stop, step=step)) # exclucive range
    xs = Vector{Float64}(undef, length(ts))
    vs = Vector{Float64}(undef, length(ts))

    xs[1] = initx
    vs[1] = initv

    for i in 1:length(ts)-1
        accbefore = acceleration(ts[i], xs[i])
        xs[i + 1] = xs[i] + vs[i] * step + accbefore * step * step / 2
        accafter = acceleration(ts[i + 1], xs[i + 1])
        vs[i + 1] = vs[i] + (accbefore + accafter) * step / 2
    end

    return ts, xs, vs
end

end
