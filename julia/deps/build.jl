# let julia handle python internally
ENV["PYTHON"] = ""

# add 
package_names = [
    "JLD",
    "GridInterpolations",
    "PyCall",
    "PyPlot"
]
for name in package_names
    Pkg.add(name)
end

# clone
urls = [
    "https://github.com/sisl/AutomotiveDrivingModels.jl.git",
    "https://github.com/sisl/AutoViz.jl.git",
    "https://github.com/sisl/BayesNets.jl.git",
    "https://github.com/sisl/NGSIM.jl.git",
    "https://github.com/sisl/AutoRisk.jl.git",
]

packages = keys(Pkg.installed())
for url in urls
    try
        id1 = search(url, "https://github.com/")[end]
        offset = search(url[(id1[end]+1):end], "/")[end]
        package = url[(id1+offset+1): (search(url,".jl.git")[1]-1)]
        if !in(package, packages)
          Pkg.clone(url)
        else
          println("$(package) already exists. Not cloning.")
        end
    catch e
        println("Exception when cloning $(url): $(e)")
    end
end

Pkg.build("AutomotiveDrivingModels")
Pkg.build("AutoViz")
Pkg.build("BayesNets")
