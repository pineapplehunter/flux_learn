---
marp: true
theme: uncover
page_number: true
# footer: 情報科学若手の会2020
paginate: true
---

# Julia で機械学習

ダニエル(@daniel_program)

<!--
Juliaで機械学習ということで発表させていただきます。
ダニエルです。よろしくお願いします。
-->

---

## 自己紹介

- ニックネーム: **ダニエル**
- 所属: **農工大 知能情報システム工学科**
- 好きなプログラミング言語: `Julia`, `Rust`

---

## みなさん Julia はご存知ですか？

---

## Julia とは

![julialang bg right 50%](https://julialang.org/assets/infra/logo.svg)

```julia
println("Hello World!")
```

- 科学計算が得意
- 書きやすい
- 速い(JIT)※

<!--
Juliaとは
-->

---

## 書きやすい一例

行列計算は簡単に使える

```julia
julia> A = [1 2
            3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> A * A
2×2 Array{Int64,2}:
  7  10
 15  22

julia> A .* A
2×2 Array{Int64,2}:
 1   4
 9  16
```

<!--
書きやすいなと感じるコードの例としては、行列計算があります。
コードを見ていただくと、1行目で行列を定義しています。縦と横に書くことで行列として認識してくれます。個人的には見て目がかなり良いと思っています。一行に書くことも可能です。
そして2つの行列の演算を載せていますが、上の方は普通の行列の積です。機械学習ではよく使う演算なので簡単に使えて便利です。
下の演算は、同じ位置の要素同士の積です。.オペレータを使うと、各要素に適用してくれます。この.オペレータは掛け算以外にも足し算や引き算割り算はもちろん、関数呼び出しにも適用できるので、なれるととても便利な文法です。
-->

---

## プロット

```julia
julia> using Plots

julia> x = 0:0.1:10
0.0:0.1:10.0

julia> y = sin.(x) # .は全てに適用
101-element Array{Float64,1}:
  0.0
  0.09983341664682815
  ⋮
 -0.3664791292519284
 -0.5440211108893698

julia> plot(x,y)
```

![Plot bg right 100%](https://raw.githubusercontent.com/pineapplehunter/flux_learn/main/sin_wave.png)

<!--
ここではデータのプロットをする例を載せています。
データのプロットは簡単にできます。ここではsin波形を出力していますが、手順としては、まずPlotsライブラリをインポートします。次にxとyを定義します。xには0から0.1刻みで10までの配列を入れています。yでは.オペレータで各x要素にsinを適用しています。

プロットには最後にplot関数を呼び出します。

このように簡単にデータのプロットもできます。
-->

---

## 速い

![Julia Micro-Benchmarks stats](https://julialang.org/assets/benchmarks/benchmarks.svg)
[Julia Micro-Benchmarks](https://julialang.org/benchmarks/)

<!--
次にJuliaの速さについて紹介します。
このグラフはいくつかの言語でマイクロベンチマークを動かした結果のグラフです。一番左がC言語で、その隣がJuliaとなっています。見てわかるとおり相当早いです。
機械学習は最近ほとんどPythonでされていますが、縦軸が対数スケールになっていることを考慮すると、Pythonと比べるとほとんどのベンチマークでかなりの差が出ていることが解ると思います。
-->

---

## Julia での機械学習はここが良い

- 簡単に記述できる(Pytorch 風)
- 自動微分が高性能
- よりインタラクティブに分析できる

<!--
さて、そんなJuliaなんですが、今までに紹介したJuliaの便利な機能の他にもJuliaが機械学習に向いている理由があります。
-->

---

![Flux.jl bg 60%](https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/logo.png)

---

## 簡単に記述できる

モデルの作り方

```julia
julia> using Flux

julia> model = Chain(
           Dense(784, 128, relu),
           Dense(128, 10),
           logsoftmax,
        )
Chain(Dense(784, 128, relu), Dense(128, 10), logsoftmax)
```

---

## モデルの使い方

```julia
julia> model(rand(784,5)) # ランダムなデータを5つ入力
10×5 Array{Float32,2}:
 -2.0062   -3.1991   -3.16117  -3.63618  -2.62927
 -3.19613  -2.61039  -2.42026  -2.23364  -2.3462
 -2.49663  -1.99805  -2.24729  -2.23044  -2.43747
 -3.3243   -2.53268  -2.64656  -2.75284  -2.51889
  ⋮
 -2.40612  -2.15469  -2.11569  -2.35603  -2.74064
 -2.17706  -2.36897  -2.75496  -2.08347  -2.19851
 -3.70686  -3.23972  -3.1855   -3.24699  -2.74348
```

学習については Pytorch と似たような記述方法になっています。

---

## 自動微分

```julia
julia> f(x) = 2x + 1 # 関数定義
f (generic function with 1 method)
julia> f(2)
5
julia> f'(2) # f(x)を微分して2を代入して計算, f'(x) = 2
2
julia> @code_llvm f'(2) # LLVMの表現を見る
define i64 @"julia_#43_13375"(i64) {
top:
  ret i64 2
}
```

多変数関数のためにはより汎用な`gradient`関数が用意されています。

<!-- LLVMの部分はコメント部分を省略しています
このように自動微分が恐ろしく簡単にできてしまうため自分のレイヤーを簡単に定義することができます。
 -->

---

## 自分の活性化関数を定義

```julia
julia> Chain(
           Dense(784, 128),
           x -> max.(0, x), # ReLUを定義しています
           Dense(128, 10),
           logsoftmax
       )

julia> model(rand(784,2))
10×2 Array{Float32,2}:
 -2.36733  -3.14914
 -1.59384  -2.54492
  ⋮
```

<!--
例えば、例として活性化関数を定義してみます。このソースコードのコメントの書いてある行を見ると、おもむろに無名関数が書いてあります。これはReLUの定義を書いています。このように雑に書いたとしてもモデルとして正しく機能します。
-->

---

## 自分のレイヤーを定義

```julia
struct MyDense # レイヤーの構造体定義
    W # 重み
    B # バイアス
end

# 入出力の個数から初期化できるようにする
MyDense(input::Integer, output::Integer) =
    MyDense(randn(output, input), randn(output))

# レイヤーの動作の定義
(m::MyDense)(x) = m.W * x .+ m.B

Flux.@functor MyDense # パラメータを学習可能にする
```

これだけの定義で Flux のモデルに組み込めます。

<!--
また、これはDenseレイヤーを自作してみています。
はじめに、MyDenseという構造体を定義しています。中身は重みのWとバイアスのBです。
次に、よく見る入力数と出力数からWとBを初期化するためのコードを書いていきます。inputとouptutはInteger型にしています。そして、中身ではランダムな値で初期化しています。
次に、レイヤーの動作を定義しています。少し不思議な構文ですが、これは構造体を関数のように呼び出したときの定義です。PythonのClassでいう__call__のようなものです。今回はW*x+bで定義しています。
最後にFluxが定義している@functorマクロを使い構造体の中の重みとバイアスを学習可能だとマークします。

これで自前のDenseレイヤーは完成です。実際のDenseレイヤーは初期化方法や活性化関数などを指定できるのでもう少しだけ文量は増えますがこれとほとんど同じ定義らしいです。
-->

---

## インタラクティブ

[![Pluto.jl](https://raw.githubusercontent.com/fonsp/Pluto.jl/master/frontend/img/logo.svg)][pluto.jl]

<!--
そして、僕がJuliaで機械学習をするメリットとして、一番推したいのがインタラクティブ性です。これは実演します。

* Julia REPL
* VSCode
* Pluto
-->

---

## 便利なライブラリ

汎用ライブラリ

- [Plots.jl][]: グラフの描写
- [PyCall.jl][]: Python の読み込み
- [DataFrames.jl][]: Pandas のようにデータを格納できる
- [CUDA.jl][]: GPU を使える

機械学習用ライブラリ

- [Zygote][]: 自動微分が実装されている
- [Metalhead][]: VGG 等のモデル
- [DifferentialEquations.jl][]: 微分方程式を解ける
- [model zoo][]: Flux で実装されたモデルいろいろ

---

## ご清聴ありがとうございました

[plots.jl]: https://github.com/JuliaPlots/Plots.jl
[pycall.jl]: https://github.com/JuliaPy/PyCall.jl
[dataframes.jl]: https://github.com/JuliaData/DataFrames.jl
[differentialequations.jl]: https://github.com/SciML/DifferentialEquations.jl
[cuda.jl]: https://github.com/JuliaGPU/CUDA.jl
[flux.jl]: https://github.com/FluxML/Flux.jl
[pluto.jl]: https://github.com/fonsp/Pluto.jl
[model zoo]: https://github.com/FluxML/model-zoo/
[metalhead]: https://github.com/FluxML/Metalhead.jl
[zygote]: https://fluxml.ai/Zygote.jl/latest/
