### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f97ae72e-ff4b-11ea-1c88-fd596094e009
begin
	let
		import Pkg
		Pkg.activate(".")
	end
	
	using Flux
	using Flux:onecold,Data.MNIST
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

# ╔═╡ adbfe39e-ffab-11ea-3ee9-cd6fb4d8416c
@bind go html"<input type='button'>" 

# ╔═╡ da41fb90-ffa7-11ea-3325-ad93acd42de4
begin
	go
	idx = rand(1:length(MNIST.images(:test)))
end

# ╔═╡ 9d3d4dca-ff53-11ea-1477-81503e06f526
img = MNIST.images(:test)[idx]

# ╔═╡ 7d186b08-ff56-11ea-3845-a9cd34915f13
img_data = reshape(Float32.(img),28,28,1,1);

# ╔═╡ 3cd1a690-ffa8-11ea-1fa8-d7570484e874
correct = MNIST.labels(:test)[idx]

# ╔═╡ 559668de-ff5c-11ea-0fe7-b1d039f5534c
predict = onecold(model(img_data),0:9)[1]

# ╔═╡ 29d61b5a-ffab-11ea-2e07-f1d22638a4c4
md"""
## $(correct == predict ? "正解" : "不正解")
"""

# ╔═╡ 76cd13b6-ffa8-11ea-34a0-e5f3d5dd20c8
md"""
正解は**$(correct)**でモデルは**$(predict)**と予想しました
"""

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
# ╠═adbfe39e-ffab-11ea-3ee9-cd6fb4d8416c
# ╠═da41fb90-ffa7-11ea-3325-ad93acd42de4
# ╠═9d3d4dca-ff53-11ea-1477-81503e06f526
# ╟─29d61b5a-ffab-11ea-2e07-f1d22638a4c4
# ╟─76cd13b6-ffa8-11ea-34a0-e5f3d5dd20c8
# ╠═7d186b08-ff56-11ea-3845-a9cd34915f13
# ╠═3cd1a690-ffa8-11ea-1fa8-d7570484e874
# ╠═559668de-ff5c-11ea-0fe7-b1d039f5534c
# ╠═c93c4e50-ff51-11ea-293f-b52d4e0ddc5a
# ╠═bc88a144-ff57-11ea-2f38-71dc1775da02
# ╟─5fc7e666-ff5e-11ea-2e0e-6fdbaa8a35e6
# ╟─22f5c920-ff55-11ea-080a-118f73dbcc87
# ╟─40b8fbbe-ff4c-11ea-259c-65677c02230a
# ╟─77145854-ff5e-11ea-1dfe-43dcb20f2397
# ╟─f97ae72e-ff4b-11ea-1c88-fd596094e009
