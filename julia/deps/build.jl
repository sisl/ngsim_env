using Pkg

# let julia handle python internally
ENV["PYTHON"] = ""

# add 
package_names = [
    "JLD",
    "GridInterpolations",
    "PyCall",
    "PyPlot",
    "HDF5"
]
for name in package_names
    Pkg.add(name)
end

# SISL packages
packages = keys(Pkg.installed())

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
    Pkg.add(PackageSpec(url="https://github.com/sisl/AutoRisk.jl.git#v0.7fixes"))
end

