using Test
using Distributions
using AutomotiveDrivingModels
using LinearAlgebra
using StatsBase # For weights function used to create weighted likelihood
using AutoViz # For SceneOverlay within visualization_functions.jl
using Interact # For @manipulate within visualization_functions.jl
using Reel

# Bring in the method definitions
include("admin_functions.jl")
include("driving_functions.jl")
include("filtering_functions.jl")
include("metrics_functions.jl")
include("visualization_functions.jl")

# Run the individual test scripts
include("test_admin_functions.jl")
include("test_driving_functions.jl")
include("test_filtering_functions.jl")
include("test_visualization_functions.jl")
