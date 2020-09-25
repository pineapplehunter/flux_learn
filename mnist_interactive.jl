### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ f97ae72e-ff4b-11ea-1c88-fd596094e009
begin
	let
		import Pkg
		Pkg.activate(".")
	end
	
	using Flux
	using Flux:onecold
 	using Plots
	using BSON
	
	# a hack to make this work!
	# issue https://github.com/fonsp/Pluto.jl/issues/301
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	
	md"initialization"
end

# ╔═╡ 02ba2850-ff4d-11ea-139f-3d5a67ea5351
model = BSON.load("model.bson")[:model]

# ╔═╡ 9d3d4dca-ff53-11ea-1477-81503e06f526
img = rand(Flux.Data.MNIST.images(:test))

# ╔═╡ 7d186b08-ff56-11ea-3845-a9cd34915f13
img_data = reshape(Float32.(img),28,28,1,1);

# ╔═╡ 559668de-ff5c-11ea-0fe7-b1d039f5534c
onecold(model(img_data),0:9)[1]

# ╔═╡ 5fc7e666-ff5e-11ea-2e0e-6fdbaa8a35e6
md"""
## Function Definitions
"""

# ╔═╡ 22f5c920-ff55-11ea-080a-118f73dbcc87
function convolve(img, c)
	output = zeros(Float32,size(img).-size(c).+ (1,1)...)
	for i in 1:size(output)[1],j in 1:size(output)[2]
		for ci in 0:size(c)[1]-1,cj in 0:size(c)[2]-1
			output[i,j] += img[i+ci,j+cj] * c[size(c).-(ci,cj)...]
		end
	end
	output
end

# ╔═╡ 40b8fbbe-ff4c-11ea-259c-65677c02230a
function layer2color(layer)
	max_val = max(layer...)
	min_val = min(layer...)
	max_size = max_val > -min_val ? max_val : -min_val
	data = layer ./ max_size
	function val2color(v)
		RGB(max(v,0),0,max(-v,0))
	end
	val2color.(data)
end

# ╔═╡ c93c4e50-ff51-11ea-293f-b52d4e0ddc5a
layer2color.([model[1].weight[:,:,1,i] for i in 1:8])

# ╔═╡ bc88a144-ff57-11ea-2f38-71dc1775da02
[layer2color(model[1](img_data)[:,:,i,1]) for i in 1:8]

# ╔═╡ 77145854-ff5e-11ea-1dfe-43dcb20f2397
md"""
## Initialization
"""

# ╔═╡ Cell order:
# ╠═02ba2850-ff4d-11ea-139f-3d5a67ea5351
# ╠═9d3d4dca-ff53-11ea-1477-81503e06f526
# ╠═7d186b08-ff56-11ea-3845-a9cd34915f13
# ╠═559668de-ff5c-11ea-0fe7-b1d039f5534c
# ╠═c93c4e50-ff51-11ea-293f-b52d4e0ddc5a
# ╠═bc88a144-ff57-11ea-2f38-71dc1775da02
# ╟─5fc7e666-ff5e-11ea-2e0e-6fdbaa8a35e6
# ╟─22f5c920-ff55-11ea-080a-118f73dbcc87
# ╟─40b8fbbe-ff4c-11ea-259c-65677c02230a
# ╟─77145854-ff5e-11ea-1dfe-43dcb20f2397
# ╟─f97ae72e-ff4b-11ea-1c88-fd596094e009
