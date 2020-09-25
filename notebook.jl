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

# ╔═╡ 23fed19c-fbeb-11ea-067b-db98c7c4fb36
begin
	import Pkg
	pkgs = ["Flux","Plots","PlutoUI","Zygote"]
	pkgs .|> Pkg.add
	using Flux,Plots,PlutoUI,Zygote
	gr()
end

# ╔═╡ 272987cc-fbeb-11ea-2a58-630dc416b566
md"# Flux Learn"

# ╔═╡ 11004ea4-fbeb-11ea-3983-691c04f861cd
md"## initialize packages"

# ╔═╡ b66c3740-fbeb-11ea-0eb9-1b1cc527a87a
@bind r Slider(1:0.1:100)

# ╔═╡ efedfc4e-fbeb-11ea-0b87-29acbc7d18b3
f(x) = x^2

# ╔═╡ 069920a0-fbec-11ea-2ff1-25b6f92e0beb
begin
	plot(f, label="f(x)", xlim=(-r,r))
	plot!(f',label="f'(x)",xlim=(-r,r))
	plot!(title="f(x) and its derivitive")
	plot!(xlabel="x",ylabel="y")
end

# ╔═╡ d0faed0a-fbef-11ea-3f22-c75dc68d9cb2
function create_model()
	h = 5
	initb = randn
	activation = tanh
	Chain(
		Dense(1,h,activation; initb=initb),
# 		Dense(h,h,activation; initb=initb),
		Dense(h,1)
	)
end

# ╔═╡ 1b918074-fbf0-11ea-0590-653c53253de3
m = create_model()

# ╔═╡ a7066690-fbf0-11ea-0f35-8f20f01c28a4
begin
	x = range(-10,10,length=1000)
	y = m(reshape(x,1,length(x)))'
	plot(x,y)
end

# ╔═╡ Cell order:
# ╟─272987cc-fbeb-11ea-2a58-630dc416b566
# ╟─11004ea4-fbeb-11ea-3983-691c04f861cd
# ╠═23fed19c-fbeb-11ea-067b-db98c7c4fb36
# ╠═b66c3740-fbeb-11ea-0eb9-1b1cc527a87a
# ╠═efedfc4e-fbeb-11ea-0b87-29acbc7d18b3
# ╠═069920a0-fbec-11ea-2ff1-25b6f92e0beb
# ╠═d0faed0a-fbef-11ea-3f22-c75dc68d9cb2
# ╠═1b918074-fbf0-11ea-0590-653c53253de3
# ╠═a7066690-fbf0-11ea-0f35-8f20f01c28a4
