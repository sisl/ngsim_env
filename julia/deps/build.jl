using Pkg

# let julia handle python internally
#ENV["PYTHON"] = ""

# What has already been installed
packages = keys(Pkg.installed())

# add 
package_names = [
    "FileIO",
    "JLD2",
    "GridInterpolations",
    "PyCall",
    "PyPlot",
    "HDF5",
    "LinearAlgebra"
]
for name in package_names
    if !in(name, packages)
        Pkg.add(name)
    end
end

# SISL packages

if !in("Vec", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/Vec.jl.git"))
end
if !in("Records", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/Records.jl.git"))
end
if !in("NGSIM", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/NGSIM.jl.git"))
end
if !in("BayesNets", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/BayesNets.jl.git"))
end
if !in("AutomotiveDrivingModels", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveDrivingModels.jl.git"))
end
if !in("AutoViz", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/AutoViz.jl.git"))
end
if !in("AutoRisk", packages)
    Pkg.add(PackageSpec(url="https://github.com/sisl/AutoRisk.jl.git", rev="v0.7fixes"))
end

