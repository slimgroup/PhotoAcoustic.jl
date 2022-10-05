using Documenter, PhotoAcoustic

makedocs(sitename="Photoacoustic imaging in Julia",
         doctest=false, clean=true,
         authors="Rafael Orozco and Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "Tutorials" => ["LearnedPrior.md"]
         ])

deploydocs(repo="github.com/slimgroup/PhotoAcoustic.jl", devbranch="main")