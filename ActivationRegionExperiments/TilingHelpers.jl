using Plots
using LazySets
using Convex
using Distributions
using SCS
using JuMP

# An object which includes a region and its affine mapping
@with_kw mutable struct AffinePolytopeRegion
    domain::AbstractPolytope
    A::Array{Float64, 2}
    b::Array{Float64, 1}
end

@with_kw mutable struct AffineAndPoint
    point::Array{Float64, 1}
    A::Array{Float64, 2}
    b::Array{Float64, 1}
end

function dist_to_set(point, set)
    A, b = tosimplehrep(set)
    dims = size(A, 2)

    # Create a problem to find the distance
    x = Variable(dims)
    #problem = minimize(sumsquares(point - x), [A*x <= b])
    problem = minimize(norm(point - x, 2), [A*x <= b])
    solve!(problem, SCS.Optimizer)
    println("Status: ", problem.status)
    return problem.optval
end

# Slide 8-3: https://web.stanford.edu/class/ee364a/lectures/geom.pdf
function max_vol_inscribed_ellipsoid(polytope)
    # Get info from the polytope
    A, b = tosimplehrep(polytope)
    dims = size(A, 2)
    num_constraints = size(A, 1)
    
    # Define the variables
    B = Semidefinite(dims) # figure out how to restirct PD instead of PSD
    d = Variable(dims)

    # Add the objective
    problem = maximize(logdet(B))
    problem.constraints += B ⪰ 1e-5 * Matrix(I, dims, dims) # ensure PD?

    # Build up the constraints
    for i = 1:num_constraints
        problem.constraints += norm(B*A[i, :], 2) + A[i, :]' * d ≤ b[i]
    end

    # Create the problem
    solve!(problem, SCS.Optimizer)
    println("Ellipsoid status: ", problem.status)
    println(vec(d.value))
    println(B.value)
    return Ellipsoid(vec(d.value), B.value^2, check_posdef = false) # It is rejecting posdef ones
end
max_vol_inscribed_ellipsoids(polytopes) = [max_vol_inscribed_ellipsoid(polytope) for polytope in polytopes]

# Some algorithm to find the center of a polytope
function polytope_center(polytope)
    ellipsoid = max_vol_inscribed_ellipsoid(polytope)
    return ellipsoid.center
end
polytope_centers(polytopes) = [polytope_center(polytope) for polytope in polytopes]
polytope_centers(affine_polytope_regions::Vector{AffinePolytopeRegion}) = polytope_centers([region.domain for region in affine_polytope_regions])

# (x2 - x1)' x ≤ 1/2 ||x2 - x1||^2
function halfspace_between(point1, point2)
    a = point2 .- point1
    b = 0.5 * a'*a + a'*point1
    return HalfSpace(a, b)
end

# Decompose a space 
function voronoi_decomposition(points; domain=Universe(length(points[1])))
    polytopes = Array{AbstractPolytope}(undef, length(points))
    # Make a polytope for each point
    for i = 1:length(points)
        halfspaces = Array{HalfSpace{Float64, Array{Float64, 1}}, 1}()
        # Each other point contributes a halfspace
        for j = 1:length(points)
            i == j || push!(halfspaces, halfspace_between(points[i], points[j])) # instead of pushing constraints, could intersect at each point and maybe it trims?
        end
        polytopes[i] = intersection(HPolytope(halfspaces), domain) # intersect each polytope with the domain
    end
    return polytopes
end

polytopes_to_decomposition(polytopes, domain) = voronoi_decomposition(polytope_centers(polytopes); domain=domain)

function plot_decomposition(polytopes, domain; outfile="./temp.png")
    plot(domain)
    decomposition = polytopes_to_decomposition(polytopes, domain)
    ellipsoids = max_vol_inscribed_ellipsoids(polytopes) # repeats computation but just for visualization
    for (polytope, ellipsoid, outer_cell) in zip(polytopes, ellipsoids, decomposition)
        plot!(polytope)
        plot!(outer_cell)
        plot!(ellipsoid)
        scatter!([ellipsoid.center[1]], [ellipsoid.center[2]])
    end
    savefig(outfile)
end

function disjoint_polytopes(n)
    polytopes = [rand(VPolygon)]
    while length(polytopes) < n
        new_poly = rand(VPolygon)
        # If it doesn't intersect with any existing ones, push it
        if all(isempty(intersection(new_poly, polytope)) for polytope in polytopes)
            push!(polytopes, new_poly)
        end
    end
    return polytopes
end