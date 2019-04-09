# Trying out the tests from separate files strategy
using Test
using Distributions
using AutomotiveDrivingModels
using LinearAlgebra
using StatsBase

# Bring in the method definitions
include("admin_functions.jl")
include("driving_functions.jl")
include("filtering_functions.jl")
include("metrics_functions.jl")

# Run the individual test scripts
include("test_admin_functions.jl")
include("test_driving_functions.jl")
include("test_filtering_functions.jl")
include("test_metrics_functions.jl")
