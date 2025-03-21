# Following  https://juliaintervals.github.io/TaylorModels.jl/latest/

using TaylorModels
using Plots

order = 4
c = 4.0
f(x) = sin(x)
# f(x) = 1 / (1 + exp(-x))
f(x) = x^3 + x^2 + x + 1
g(x) = (f(x) - f(c))^2


# Define domain and expansion point
a = c-5.0 .. c+5.0

# Create Taylor models
tm = TaylorModel1(order, interval(c), a)

# Compute Taylor models of g(x)
ftm = g(tm)

# Plotting
gr()
plot(range(inf(a), stop=sup(a), length=100), x->g(x), 
    label="true", lw=2, xaxis="x", yaxis="||f(x)-f(c)||^2", 
    title="f(x) = x^3 + x^2 + x + 1, order=3, c= 4.0")
plot!(ftm, label="taylor model")



#=
# Define the partition of [-5,5] into 10 intervals
num_partitions = 10
intervals = [(-5 + i * 10 / num_partitions) .. (-5 + (i + 1) * 10 / num_partitions) for i in 0:num_partitions-1]

# Create plot
gr()
plot(title="f(x) = sin(x)", xaxis="x", yaxis="||f(x)-f(c)||^2", legend=:topright)

for a in intervals
    # Create Taylor model
    tm = TaylorModel1(order, interval(mid(a)), a)
    
    # Compute Taylor model of g(x)
    ftm = g(tm)
    
    # Plot Taylor model
    plot!(ftm, label=nothing)  # Remove label to avoid clutter
end

# Plot the true function for reference
x_vals = range(-5, stop=5, length=200)
plot!(x_vals, x -> g(x), label="true", lw=2)
=#