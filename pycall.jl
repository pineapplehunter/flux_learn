### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 860fda80-fece-11ea-0a5b-53e40e2718ff
begin
	import Pkg; Pkg.activate(mktempdir())
	pkgs = ["PyCall"]
	pkgs .|> Pkg.add
	using PyCall
end

# ╔═╡ 0892506e-fecf-11ea-24a2-4b0005de04ad
np = pyimport("numpy")

# ╔═╡ 12c4696e-fecf-11ea-270d-310902a1bc68
np.array([1,2,3,4])

# ╔═╡ Cell order:
# ╠═860fda80-fece-11ea-0a5b-53e40e2718ff
# ╠═0892506e-fecf-11ea-24a2-4b0005de04ad
# ╠═12c4696e-fecf-11ea-270d-310902a1bc68
