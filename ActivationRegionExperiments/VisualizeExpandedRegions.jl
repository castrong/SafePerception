using Plots
using LazySets
using Convex
using Distributions
using SCS
using JuMP

# Include all our helper functions
include("./TilingHelpers.jl")

# Sample points in the hyperrectangle between 0 and 1.
dims = 2
num_points = 1
distr = Product(Uniform.(zeros(dims), 2 .* ones(dims)))
points = [rand(distr, 1) for i = 1:num_points]

polytope1 = HPolytope([3.0 1.0; -1.0 -1.0; -1.0 1.0; 1.0 -1.0], [1.0; 1.0; 1.0; 1.0])
#polytope2 = Hyperrectangle(low=[2.0, 2.0], high=[4.0, 4.0])
polytope2 = HPolytope([1.0 1.0; -3.0 -1.0; -3.0 2.0; 1.0 -2.0], [6.0; 8.0; -2.0; -1.0])
# rand(VPolygon)
plot(polytope1)
plot!(polytope2)

# Fill this with 1 if the point is closer to shape 1, 
# 0 if it's closer to shape 2
closer_to_one = []
for point in points
    distance_one = dist_to_set(point, polytope1)
    distance_two = dist_to_set(point, polytope2)
    push!(closer_to_one, distance_one < distance_two ? 1 : 0)
end

points_closer_one = points[closer_to_one .== 1]
points_closer_two = points[closer_to_one .== 0]

# Find the maximum volume inscribed ellipsoids
ellipsoid1 = max_vol_inscribed_ellipsoid(polytope1)
ellipsoid2 = max_vol_inscribed_ellipsoid(polytope2)

# Find the halfspace separating them
halfspace = halfspace_between(ellipsoid1.center, ellipsoid2.center)

# Plot the polytopes
plot(polytope1; aspect_ratio=:equal, title="L-2 norm regions", legend=:none)
plot!(polytope2)

# Plot the ellipsoids and their centers
plot!(ellipsoid1)
scatter!([ellipsoid1.center[1]], [ellipsoid1.center[2]])
plot!(ellipsoid2)
scatter!([ellipsoid2.center[1]], [ellipsoid2.center[2]])

# Plot a line connecting the centers
plot!([ellipsoid1.center[1], ellipsoid2.center[1]], [ellipsoid1.center[2], ellipsoid2.center[2]])

scatter!([x[1] for x in points_closer_one], [x[2] for x in points_closer_one], color=:blue, markersize=0.1)
scatter!([x[1] for x in points_closer_two], [x[2] for x in points_closer_two], color=:red, markersize=0.1)

plot!(halfspace)

