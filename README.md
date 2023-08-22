<link rel="stylesheet" href="imagens/style.css">

<h2 id="inicio">Códigos, algoritmos, exemplos e aplicações</h2>

<p>Esta página contém os códigos, algoritmos e exemplos das técnicas mostradas na disciplina de Visualização Científica.</p>
<p>A apostila está disponível no link: <a href="imagens/apostila_2022.pdf" target="_blank">apostila de Visualização Científica</a></p>

<details>
  <summary id="modulo1">1. Introdução</summary>
  <p>Material da página 1 até a página 14.</p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-0.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/9wgs_H7FCzM" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
   <img src="modulo1/59f0152f9f78F561f6fb413c7e4f88ba0-1.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-2.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-3.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-4.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-5.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-6.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-7.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-8.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-9.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-10.png" loading="lazy"/>
   <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/aOVwjCVWx1w" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-10a.png" loading="lazy"/>   
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-11.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-12.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-13.png" loading="lazy"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo2">2. Conceitos básicos e estruturais de visualização</summary>
  <p>Material da página 14 até a página 24.</p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-13.png" loading="lazy"/>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/hH89iocE484" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div> 
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod1', 'cd1')" onmouseout="outFunc('cd1')"><span class="tooltiptext" id="cd1">Copiar o código</span></button></div>Código em C++ com Matplotlib:
<pre><code id="cod1">#include <a alt="vetores de coordenadas">&lt;vector&gt;</a> 
&#x23;include <a alt="biblioteca matplotlib">"matplotlibcpp.h"</a> 
Namespaceplt <a alt="gráfico que será construído">plt=matplotlibcpp;</a> 

intmain()&#123;
std::vector&lt;double&gt;<a alt="coordenadas x">x=&#123;0, 1, 2, 3, 4, 5&#125;;</a>
std::vector&lt;double&gt;<a alt="coordenadas y">y=&#123;1, 4, 9, 16, 32, 64&#125;;</a>
<a alt="gráfico de dispersão 2D, marcador circular e vermelho">plt::scatter(x,y,{&#125;"color","red"&#123;,&#123;"marker":"o"&#125;&#125;);</a>
<a alt="comando para visualizar o gráfico">plt::show();</a>

return0;
&#125;

</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod2', 'cd2')" onmouseout="outFunc('cd2')"><span class="tooltiptext" id="cd2">Copiar o código</span></button></div>Código em Python com Matplotlib:
<pre><code id="cod2">import <a alt="gráfico plt da biblioteca matplotlib ">matplotlib.pyplot as plt</a> 

<a alt="coordenadas x">x =</a> [0, 1, 2, 3, 4, 5]
<a alt="coordenadas y">y =</a> [1, 4, 9, 16, 32, 64]

<a alt="gráfico de dispersão 2D">plt.scatter</a>(x, y, <a alt="marcador vermelho">color =</a> 'red', <a alt="marcador circular">marker =</a> 'o')
<a alt="comando para visualizar o gráfico">plt.show()</a>

</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-15.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod3', 'cd3')" onmouseout="outFunc('cd3')"><span class="tooltiptext" id="cd3">Copiar o código</span></button></div>Gráfico de dispersão 3D:
<pre><code id="cod3">import <a alt="gráfico plt da biblioteca matplotlib ">matplotlib.pyplot as plt</a> 

<a alt="coordenadas x">x =</a> [0, 1, 2, 3, 4, 5]
<a alt="coordenadas y">y =</a> [1, 4, 9, 16, 32, 64]
<a alt="coordenadas z">z =</a> [2, 7, 11, 5, 3, 1]

<a alt="tipo de projeção 3D; gráfico atribuído na variável ax">ax =</a> plt.figure().add_subplot(projection = '3d')

<a alt="gráfico de dispersão 3D, marcador circular e vermelho">ax.scatter(x, y,</a> z, color = 'r', marker = 'o')

plt.show()

</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-15a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-16.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://www.w3schools.com/colors/colors_rgb.asp" target="_blank">https://www.w3schools.com/colors/colors_rgb.asp</a></p></details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-16a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-17.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Links</summary>
  <p><a href="https://www.w3schools.com/colors/colors_names.asp" target="_blank">https://www.w3schools.com/colors/colors_names.asp</a></p>
  <p><a href="https://matplotlib.org/stable/gallery/color/named_colors.html" target="_blank">https://matplotlib.org/stable/gallery/color/named_colors.html</a></p></details></div>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/2gt9w7yPI7g" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-17a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-18.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod4', 'cd4')" onmouseout="outFunc('cd4')"><span class="tooltiptext" id="cd4">Copiar o código</span></button></div>Gráfico de dispersão 2D com rótulos:
<pre><code id="cod4">import matplotlib.pyplot as plt 

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 9, 16, 32, 64, 128]
<a alt="rótulos dos pontos">rotulos =</a> ['A', 'B', 'C', 'D', 'E', 'F', 'G']

<a alt="laço para rotular cada ponto">for i, txt in enumerate(rotulos):</a>
    plt.annotate(txt, (x[i], y[i]))
	
plt.plot(x, y, <a alt="marcador laranja">color =</a> 'orange', <a alt="marcador triângular">marker =</a> '^', <a alt="linha contínua">linestyle =</a> '-')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-18a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-19.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod5', 'cd5')" onmouseout="outFunc('cd5')"><span class="tooltiptext" id="cd5">Copiar o código</span></button></div>Gráfico de dispersão 3D com rótulos:
<pre><code id="cod5">import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection = '3d')

x = [0, 1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 32, 64]
z = [2, 7, 11, 5, 3, 1]
<a alt="rótulos dos pontos">rotulos =</a> ['A', 'B', 'C', 'D', 'E', 'F']

<a alt="gráfico de dispersão 3D">ax.scatter(x, y,</a> z, color = 'r', marker = 'o')

<a alt="laço para rotular cada ponto">for x, y, z, label in zip(x, y, z, rotulos):</a>
    ax.text3D(x, y, z, label, <a alt="direção dos rótulos: eixo z">zdir = 'z'</a>)
	
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-19a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod6', 'cd6')" onmouseout="outFunc('cd6')"><span class="tooltiptext" id="cd6">Copiar o código</span></button></div>Gráfico de curvas 2D com legendas:
<pre><code id="cod6">import matplotlib.pyplot as plt
import <a alt="biblioteca de operações matemáticas">numpy as np</a>

<a alt="intervalo [0, 5] com espaçamento 0.1">x = np.arange(0, 5, 0.1)</a>

<a alt="função linear">plt.plot</a>(x, x, <a alt="linha tracejada azul">'b--',</a> label = 'y = x')
<a alt="função linear">plt.plot</a>(x, 2*x+1, <a alt="linha contínua verde">'g-',</a> label = 'y = 2x + 1')
<a alt="função quadrática">plt.plot</a>(x, x**2+2*x+3, <a alt="linha traço-ponto vermelha">'r-.',</a> label = 'y = x^2 + 2x + 3')

<a alt="rótulo do eixo x">plt.xlabel('x')</a>
<a alt="rótulo do eixo y">plt.ylabel('y')</a>
<a alt="título do gráfico">plt.title('Gráfico de curvas 2D')</a>

plt.show()
plt.legend()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-20.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod7', 'cd7')" onmouseout="outFunc('cd7')"><span class="tooltiptext" id="cd7">Copiar o código</span></button></div>Gráficos de curvas 2D:
<pre><code id="cod7">import matplotlib.pyplot as plt
import numpy as np

<a alt="definição da função f">def f(x):</a>
    return np.exp(-x) * np.cos(2*np.pi*x)

<a alt="intervalo do primeiro gráfico">x1 =</a> np.arange(5, 12, 0.05)
<a alt="intervalo do segundo gráfico">x2 =</a> np.arange(-2, 5, 0.05)

plt.figure()
<a alt="1 linha e 2 colunas de gráficos: gráfico 121">plt.subplot(121)</a>
plt.plot(x1, f(x1), <a alt="linha tracejada azul no intervalo x1">'b--',</a> x2, f(x2), <a alt="traço e ponto verde no intervalo x2">'g-.')</a>

<a alt="1 linha e 2 colunas de gráficos: gráfico 122">plt.subplot(122)</a>
plt.plot(x2, np.cos(2*np.pi*x2), <a alt="linha pontilhada laranja no intervalo x2">color = 'orange', linestyle = ':')</a>
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-20a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-21.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod8', 'cd8')" onmouseout="outFunc('cd8')"><span class="tooltiptext" id="cd8">Copiar o código</span></button></div>Gráficos de curvas 3D:
<pre><code id="cod8">import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

int = 10
<a alt="intervalo de x com 10 pontos">x =</a> np.linspace(-5, 5, int)
<a alt="intervalo de y com 10 pontos">y =</a> np.linspace(-5, 5, int)
<a alt="intervalo de z com 10 pontos">z =</a> np.linspace(-10, 10, int)

<a alt="gráfico da função quadrática com marcadores circulares vermelhos">ax.plot(x, y, z**2+5, 'ro-')</a>
<a alt="gráfico da função linear com linha contínua laranja">ax.plot(x, y, 0, 'y--')</a>

plt.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-22.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod9', 'cd9')" onmouseout="outFunc('cd9')"><span class="tooltiptext" id="cd9">Copiar o código</span></button></div>Gráfico da hélice cilíndrica:
<pre><code id="cod9">import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

d = 5
<a alt="grid para a variável z">z =</a> np.linspace(-10, 10, 100)
<a alt="equação com parâmetro d para x">x =</a> d * np.sin(z)
<a alt="equação com parâmetro d para y">y =</a> d * np.cos(z)

<a alt="gráfico da hélice cilíndrica com linha contínua verde">ax.plot(x, y, z, 'g-', label = 'hélice cilíndrica')</a>
ax.legend()

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-22a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-23.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod10', 'cd10')" onmouseout="outFunc('cd10')"><span class="tooltiptext" id="cd10">Copiar o código</span></button></div>Gráfico da hélice cilíndrica com segmentos projetantes:
<pre><code id="cod10">import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

d = 5
z = np.linspace(0, 2*np.pi, 25)
x = d * np.sin(z)
y = d * np.cos(z)

<a alt="gráfico da hélice cilíndrica com segmentos projetantes">ax.stem(x, y, z)</a>

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-23a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo3">3. Fundamentos dos dados</summary>
  <p>Material da página 24 até a página 54.</p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-23.png" loading="lazy"/>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/3SN9-o3iP2c" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div> 
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-24.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-25.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Links</summary>
  <p>Conjunto Iris: <a href="https://archive.ics.uci.edu/ml/datasets/iris" target="_blank">https://archive.ics.uci.edu/ml/datasets/iris</a>
  <br>Conjunto dos Pinguins: <a href="https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html" target="_blank">https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html</a>
  <br>Outros conjuntos de dados: <a href="https://www.maptive.com/free-data-visualization-data-sets/" target="_blank">https://www.maptive.com/free-data-visualization-data-sets/</a>
  </p></details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-25a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-26.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod11', 'cd11')" onmouseout="outFunc('cd11')"><span class="tooltiptext" id="cd11">Copiar o código</span></button></div>Conjunto de dados Iris com matplotlib (cor e movimento):
<pre><code id="cod11"><a alt="biblioteca para leitura dos dados em formato CSV">import pandas as pd</a>
import numpy as np
from matplotlib import pyplot as plt

iris = <a alt="leitura do arquivo CSV">pd.read_csv('C:/dados/iris.csv')</a>

<a alt="variável usada para contar o número de registros">incl = np.array(iris.loc[:,'Largura da Sépala'])</a>
<a alt="variável x">x =</a> np.array(iris.loc[:,'Comprimento da Sépala'])
<a alt="variável y">y =</a> np.array(iris.loc[:,'Comprimento da Pétala'])
<a alt="atributo usado para separação dos dados">z =</a> np.array(iris.loc[:,'Espécie'])

<a alt="criação da grade com a quantidade de registros">i =</a> np.arange(0, len(incl), 1)
<a alt="ajuste dos dados usando os limites max e min">j =</a> (incl - min(incl))/(max(incl) - min(incl))
<a alt="amplitude máxima de 45&deg;">w =</a> -45*j

<a alt="rótulos dos dados com as espécies">label =</a> ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
<a alt="vetor com a cor de cada espécie">cor =</a> ['orange', 'green', 'red']
j = 0

<a alt="laço usado para separar as espécies">for k in i:</a>
    <a alt="marcador de cada registro">marker =</a> (2, 1, w[k])
    if <a alt="com os dados ordenados, o marcador muda quando z[k] &ne; z[k-1]">z[k] == z[k-1]:</a>
        plt.plot(x[k], y[k], <a alt="marcadores">marker =</a> marker, <a alt="tamanho dos marcadores">markersize =</a> 10, <a alt="cores dos marcadores">color =</a> cor[j - 1], <a alt="opacidade">alpha =</a> 0.6)
    else:
        j +=1
        plt.plot(x[k], y[k], marker = marker, markersize = 10, color = cor[j - 1], alpha = 0.6, 
        <a alt="alteração do rótulo">label =</a> label[j - 1] )

<a alt="legendas dos dados e eixos">plt.legend(scatterpoints = 1)</a>
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Comprimento da Pétala')
plt.grid()
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-26a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-27.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod12', 'cd12')" onmouseout="outFunc('cd12')"><span class="tooltiptext" id="cd12">Copiar o código</span></button></div>Conjunto de dados Iris com matplotlib (textura):
<pre><code id="cod12">import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

iris = pd.read_csv('C:/dados/iris.csv')

x = np.array(iris.loc[:,'Comprimento da Sépala'])
y = np.array(iris.loc[:,'Comprimento da Pétala'])
z = np.array(iris.loc[:,'Espécie'])

i = np.arange(0, len(x), 1)

label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
<a alt="textura de cada espécie">tex =</a> ['\\', '.', '+']
<a alt="marcadores de cada espécie">marker =</a> ['^', 'X', 'o']
j = 0

<a alt="laço usado para separar as espécies">for k in i:</a>
    if <a alt="com os dados ordenados, o marcador muda quando z[k] &ne; z[k-1]">z[k] == z[k-1]:</a>
        plt.scatter(x[k], y[k], <a alt="marcadores">marker =</a> marker[j - 1], <a alt="tamanho dos marcadores">s =</a> 200, <a alt="cor dos marcadores">facecolor =</a> 'white', 
        <a alt="textura">hatch =</a> 5*tex[j-1], alpha = 0.5)
    else:
        j += 1
        plt.scatter(x[k], y[k], marker = marker[j - 1], s = 200, facecolor = 'white', 
        hatch = 5*tex[j-1], alpha = 0.5, label = label[j-1])

plt.xlabel('Comprimento da Sépala')
plt.ylabel('Comprimento da Pétala')
plt.legend(scatterpoints = 1)
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-27a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-28.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod13', 'cd13')" onmouseout="outFunc('cd13')"><span class="tooltiptext" id="cd13">Copiar o código</span></button></div>Conjunto de dados Iris com Seaborn (cores):
<pre><code id="cod13">import pandas as pd
<a alt="biblioteca Seaborn">import seaborn as sns</a>

iris = pd.read_csv('C:/dados/iris.csv')

sns.relplot(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    <a alt="separação de classes: atributo Espécie">hue =</a> 'Espécie', <a alt="marcadores triangulares">marker =</a> '>', <a alt="paleta de cores em tons de azul">palette =</a> 'Blues')</a>
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-28a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/ZIa_LA38AYg" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-29.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-30.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-31.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-32.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod14', 'cd14')" onmouseout="outFunc('cd14')"><span class="tooltiptext" id="cd14">Copiar o código</span></button></div>Conjunto de dados Iris com Seaborn (cores e tamanhos):
<pre><code id="cod14">import pandas as pd
import seaborn as sns

iris = pd.read_csv('C:/dados/iris.csv')

sns.relplot(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    <a alt="separação de classes: atributo Espécie">hue =</a> 'Espécie', <a alt="marcadores quadrados">marker =</a> 's', <a alt="paleta de cores em tons de vermelho">palette =</a> 'Reds', <a alt="tamanhos dos marcadores">size =</a> 'Largura da Sépala')
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-32a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-33.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod15', 'cd15')" onmouseout="outFunc('cd15')"><span class="tooltiptext" id="cd15">Copiar o código</span></button></div>Conjunto de dados Iris com Seaborn (dispersão e frequência):
<pre><code id="cod15">import pandas as pd
import seaborn as sns

iris = pd.read_csv('C:/dados/iris.csv')

<a alt="mostra a grade">sns.set_style("whitegrid")</a>
<a alt="função de dispersão e frequência">sns.jointplot</a>(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    hue = 'Espécie', <a alt="marcadores circulares">marker =</a> 'o', <a alt="paleta de cores rainbow">palette =</a> 'rainbow')
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-33a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-34.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod16', 'cd16')" onmouseout="outFunc('cd16')"><span class="tooltiptext" id="cd16">Copiar o código</span></button></div>Conjunto de dados dos pinguins com Seaborn (regressão linear):
<pre><code id="cod16">import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
<a alt="função de dispersão e frequência">sns.lmplot</a>(data = pinguins, x = 'Comprimento do bico', y = 'Massa corporal',
    hue = 'Espécie', <a alt="paleta de cores rocket">palette =</a> 'rocket')

</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-34a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod17', 'cd17')" onmouseout="outFunc('cd17')"><span class="tooltiptext" id="cd17">Copiar o código</span></button></div>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code id="cod17">import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
<a alt="dados que devem ser desconsiderados">pinguins.drop(['Id','Ano'],</a> inplace = True, axis = 1)
<a alt="combinação de gráficos">sns.pairplot</a>(data = pinguins, hue = 'Espécie', <a alt="paleta de cores cubehelix">palette =</a> 'cubehelix')

</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod18', 'cd18')" onmouseout="outFunc('cd18')"><span class="tooltiptext" id="cd18">Copiar o código</span></button></div>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code id="cod18">import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
pinguins.drop(['Id','Ano'], inplace = True, axis = 1)
<a alt="combinação de gráficos">g = sns.PairGrid</a>(data = pinguins, hue = 'Espécie', <a alt="paleta de cores mako">palette =</a> 'mako')
<a alt="histogramas na diagonal">g.map_diag(sns.histplot)</a>
<a alt="densidades de kernel fora da diagonal">g.map_offdiag(sns.kdeplot)</a>
g.add_legend()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod19', 'cd19')" onmouseout="outFunc('cd19')"><span class="tooltiptext" id="cd19">Copiar o código</span></button></div>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code id="cod19">import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
pinguins.drop(['Id','Ano'], inplace = True, axis = 1)
<a alt="combinação de gráficos">g = sns.PairGrid</a>(data = pinguins, hue = 'Espécie', <a alt="paleta de cores mako">palette =</a> 'mako')
<a alt="histogramas na diagonal">g.map_diag(sns.histplot)</a>
<a alt="dispersão na diagonal superior">g.map_upper(sns.scatterplot, size = pinguins['Sexo'])</a>
<a alt="densidades de kernel na diagonal inferior">g.map_lower(sns.kdeplot)</a>
g.add_legend(title = '', adjust_subtitles = True)
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Exemplo do uso da técnica PCA</summary>
	<p>Técnica PCA aplicada ao conjunto de dados Iris.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="100" name="sl">
			   <label for="100"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod999', 'cd999')" onmouseout="outFunc('cd999')"><span class="tooltiptext" id="cd999">Copiar o código</span></button></div>PCA aplicada ao conjunto Iris:
<pre><code id="cod999"><a alt="importação das bibliotecas">import pandas as pd</a>
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets

import seaborn as sns

<a alt="dados Originais">iris = datasets.load_iris()</a>
X = pd.DataFrame (iris.data, 
columns = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)",])
y = pd.DataFrame (iris.target, columns = ["target"])
df = X.join (y)

<a alt="gráfico com as combinações de variáveis">sns.pairplot</a> (df, hue = "target",palette = "tab10");

</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="101" name="sl">
			   <label for="101"></label>
			   <img src="modulo3/pca1.png" loading="lazy"/>
			   <figcaption>Dados originais do conjunto Iris.</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="102" name="sl">
			   <label for="102"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod998', 'cd998')" onmouseout="outFunc('cd998')"><span class="tooltiptext" id="cd998">Copiar o código</span></button></div>PCA aplicada ao conjunto Iris:
<pre><code id="cod998"><a alt="dados modificados com o PCA">X = iris.data</a>
y = iris.target

pca = <a alt="escolha de 2 variáveis">decomposition.PCA(n_components=2)</a>
pca.fit(X)
X = pd.DataFrame (pca.transform(X), columns = ["Component 1", "Component 2"])
y = pd.DataFrame (iris.target, columns = ["target"])
df = X.join (y)

<a alt="gráfico com a combinação de variáveis obtida pelo PCA">sns.pairplot</a> (df, hue = "target",palette = "tab10");

</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="103" name="sl">
			   <label for="103"></label>
			   <img src="modulo3/pca2.png" loading="lazy"/>
			   <figcaption>Dados com a combinação de 2 variáveis do conjunto Iris com o uso da técnica PCA.</figcaption>
		   </li>
		</ul>
		<img src="modulo3/pca1.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36b.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod20', 'cd20')" onmouseout="outFunc('cd20')"><span class="tooltiptext" id="cd20">Copiar o código</span></button></div>Conjunto de dados Iris com Plotly (dispersão 3D):
<pre><code id="cod20">import pandas as pd
<a alt="biblioteca plotly">import plotly.io as pio</a>
import plotly.express as px

iris = pd.read_csv('C:/dados/iris.csv')

pio.renderers
<a alt="a renderização é feita em um navegador de internet">pio.renderers.default = 'browser'</a>

fig = <a alt="gráfico de dispersão 3D">px.scatter_3d</a>(iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala', 
    z = 'Largura da Sépala', <a alt="a cor é usada para separar as espécies">color =</a> 'Espécie')

fig.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/UigLE_aRPb0" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-37.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-38.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-39.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod21', 'cd21')" onmouseout="outFunc('cd21')"><span class="tooltiptext" id="cd21">Copiar o código</span></button></div>Duas hélices (movimento):
<pre><code id="cod21">import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

d = 1
e = 10
op = 0.9

z = np.arange(0, 200, 1)
x = d * np.sin(z/e)
y = d * np.cos(z/e)

<a alt="laço para mudar opacidade dos pontos">for k in z:</a>
    op *= 0.99
    ax.scatter(x[k], z[k], y[k], zdir = 'z', color = 'steelblue', alpha = op)
    ax.scatter(y[k] - d, d - z[k], x[k], zdir = 'z', color = 'lightcoral', alpha = op)

plt.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-40.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod22', 'cd22')" onmouseout="outFunc('cd22')"><span class="tooltiptext" id="cd22">Copiar o código</span></button></div>Dados vetoriais 2D:
<pre><code id="cod22">import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [2, 1, 2]
<a alt="deslocamento u relativo à coordenada x">U =</a> [1, -1, 0.5]
<a alt="deslocamento v relativo à coordenada y">V =</a> [1, -1, -2]
cor = ['blue','red','orange']

fig, ax = plt.subplots()
Q = <a alt="gráfico vetorial: quiver">ax.quiver</a>(X, Y, U, V, color = cor, units = 'xy', <a alt="espessura da linha e da seta">width =</a> 0.02, <a alt="escala do tamanho da seta">scale =</a> 1)

<a alt="pontos iniciais das setas">ax.plot(X, Y, 'og')</a>
ax.set_aspect('equal', 'box')

<a alt="limites dos eixos">plt.xlim([0, 4])</a>
plt.ylim([-0.5, 3.5])
plt.grid()
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod23', 'cd23')" onmouseout="outFunc('cd23')"><span class="tooltiptext" id="cd23">Copiar o código</span></button></div>Dados vetoriais 2D:
<pre><code id="cod23">import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dados = pd.read_csv('C:/dados/dados_teste.csv')

X = dados.loc[:,'x']
Y = dados.loc[:,'y']
U = dados.loc[:,'u']
V = dados.loc[:,'v']
fig, ax = plt.subplots()

<a alt="vetor usado para criar cores usando os valores U e V">M = np.hypot(U, V)</a>
Q = ax.quiver(X, Y, U, V, M, units = 'x', width = 0.07, scale = 1.2)

ax.set_aspect('equal', 'box')

plt.grid()
plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo3/dados_teste.csv" target="_blank">Arquivo CSV</a></p>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod24', 'cd24')" onmouseout="outFunc('cd24')"><span class="tooltiptext" id="cd24">Copiar o código</span></button></div>Dados vetoriais 2D:
<pre><code id="cod24">import matplotlib.pyplot as plt
import numpy as np

<a alt="extremidades na grade">X, Y =</a> np.meshgrid(np.arange(-5,5,0.5),np.arange(-5,5,0.5))

<a alt="relação do deslocamento de x com x e y">U =</a> -Y/np.hypot(X, Y)
<a alt="relação do deslocamento de y com x e y">V =</a> X/np.hypot(X, Y)

M = np.hypot(U**3, V**3)
fig, ax = plt.subplots()
Q = ax.quiver(X, Y, U, V, M, units = 'xy', width = 0.05, scale = 1.5)

ax.set_aspect('equal', 'box')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod25', 'cd25')" onmouseout="outFunc('cd25')"><span class="tooltiptext" id="cd25">Copiar o código</span></button></div>Dados vetoriais 3D:
<pre><code id="cod25">import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection = '3d')

x = [1, 2, 1]
y = [1, 2, 3]
<a alt="variável z">z =</a> [2, 1, 1]
u = [1, 2, 1]
v = [1, 1, -1]
<a alt="deslocamento da variável z">w =</a> [2, 3, -2]

Q = ax.quiver(x, y, z, u, v, w, length = 0.4, <a alt="vetores normalizados">normalize =</a> True, color = 'green', <a alt="espessura">lw =</a> 0.8)

ax.plot(x, y, z, 'ob')
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-43.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod26', 'cd26')" onmouseout="outFunc('cd26')"><span class="tooltiptext" id="cd26">Copiar o código</span></button></div>Dados vetoriais 3D:
<pre><code id="cod26">import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

<a alt="extremidades na grade">x, y, z =</a> np.meshgrid(np.arange(-2, 2, 0.5), np.arange(-2, 2, 0.5), np.arange(-2, 2, 0.5))
<a alt="relação do deslocamento de x com x e z">u =</a> x - z
<a alt="relação do deslocamento de y com y e z">v =</a> y + z
<a alt="relação do deslocamento de z">w =</a> z + 1

<a alt="relação de cores com a função arcsinh de v/u">cor =</a> np.arcsinh(v, u)
cor = (cor.flatten() - cor.min()) / cor.ptp()
<a alt="conversão de valores para o mapa de cores hsv">cor =</a> plt.cm.hsv(cor)

a = ax.quiver(x, y, z, u, v, w, length = 0.4, normalize = True, colors = cor, lw = 0.8)

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-43a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod27', 'cd27')" onmouseout="outFunc('cd27')"><span class="tooltiptext" id="cd27">Copiar o código</span></button></div>Dados vetoriais 3D:
<pre><code id="cod27">import matplotlib.pyplot as plt
import numpy as np
<a alt="biblioteca para leitura do arquivo mat">from scipy.io import loadmat</a>

ax = plt.figure().add_subplot(projection = '3d')
<a alt="leitura do arquivo de dados">data_dict =</a> loadmat('c:/dados/flow_field.mat')

x = np.transpose(data_dict['X'])
y = np.transpose(data_dict['Y'])
z = np.transpose(data_dict['Z'])
u = np.transpose(data_dict['V_x'])
v = np.transpose(data_dict['V_y'])
w = np.transpose(data_dict['V_z'])

ind = np.arange(0, len(x), 1500)

<a alt="relação de cores com a função arctan2 de u/v">c =</a> (np.arctan2(u, v))
c = (c.flatten() - c.min()) / c.ptp()
c = np.concatenate((c, np.repeat(c, 2)))
<a alt="conversão de valores para o mapa de cores hsv">c =</a> plt.cm.hsv(c)

ax.quiver(x[ind], y[ind], z[ind], u[ind], v[ind], w[ind], colors = c, length = 1.5, 
    normalize = True, lw = 0.7)

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo3/flow_field.zip" target="_blank">Arquivo MAT</a></p>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://www.bco-dmo.org/dataset/834530" target="_blank">https://www.bco-dmo.org/dataset/834530</a></p></details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44b.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv" target="_blank">https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv</a></p></details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44c.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-45.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod28', 'cd28')" onmouseout="outFunc('cd28')"><span class="tooltiptext" id="cd28">Copiar o código</span></button></div>Dados vetoriais 3D:
<pre><code id="cod28">import numpy as np
import matplotlib.pyplot as plt
<a alt="biblioteca para leitura do arquivo nc">import netCDF4 as nc</a>

<a alt="leitura do arquivo de dados">data =</a> nc.Dataset('C:/dados/tornado3d.nc')
ax = plt.figure().add_subplot(projection = '3d')

u = np.array(data['u'])
v = np.array(data['v'])
w = np.array(data['w'])
xx = np.array(data['xdim'])
yy = np.array(data['ydim'])
zz = np.array(data['zdim'])

x, y, z = np.meshgrid(xx, yy, zz)

<a alt="relação de cores com a função arctan2 de v/u">c =</a> (np.arctan2(v, u))
c = (c.flatten() - c.min()) / c.ptp()
c = np.concatenate((c, np.repeat(c, 2)))
<a alt="conversão de valores para o mapa de cores hsv">c =</a> plt.cm.hsv(c)

a = ax.quiver(x, z, y, w, u, v, colors = c, length = 0.05, normalize = True, 
    lw = 2, alpha = 0.1)

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo3/tornado3d.zip" target="_blank">Arquivo NC</a></p>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-45a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-46.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/lnIB0oqEe7I" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-47.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod29', 'cd29')" onmouseout="outFunc('cd29')"><span class="tooltiptext" id="cd29">Copiar o código</span></button></div>Linhas de fluxo 2D:
<pre><code id="cod29">import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio

pio.renderers
pio.renderers.default = 'browser'

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
Y, X = np.meshgrid(x, y)
u = 1 - X**2 + Y
v = -1 + X - Y**2
  
fig = <a alt="função para criar as linhas de fluxo: streamlines">ff.create_streamline</a>(x, y, u, v, arrow_scale = 0.05)

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-47a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod30', 'cd30')" onmouseout="outFunc('cd30')"><span class="tooltiptext" id="cd30">Copiar o código</span></button></div>Linhas de fluxo 3D:
<pre><code id="cod30">import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

pio.renderers
pio.renderers.default = 'browser'

df = pd.read_csv('C:/dados/streamtube-wind.csv')

fig = go.Figure(data = <a alt="função para criar as linhas de fluxo 3D: streamtubes">go.Streamtube</a>(x = df['x'], y = df['y'], z = df['z'], u = df['u'],
    v = df['v'], w = df['w'], sizeref = 0.3, colorscale = 'rainbow', maxdisplayed = 3000))

fig.update_layout(scene = dict(aspectratio = dict(x = 1.5, y = 1, z = 0.3)))

fig.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo3/streamtube-wind.zip" target="_blank">Arquivo CSV</a></p>
  </details></div>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-48.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-49.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Comandos básicos de dataframe</summary>
	<p>Exemplos de comandos da biblioteca pandas para dataframes.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="104" name="sl">
			   <label for="104"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod997', 'cd997')" onmouseout="outFunc('cd997')"><span class="tooltiptext" id="cd997">Copiar o código</span></button></div>Criando um dataframe:
<pre><code id="cod997"><a alt="importação das bibliotecas">import pandas as pd</a>
import numpy as np

<a alt="atributos x1, x2 e x3">x1 = </a>[1,2,1,1,3,4]
x2 = [7,8,np.nan,6,9,8]
x3 = [6,6,4,3,9,7]
<a alt="rótulos de classificação">classificacao =</a> ['Tipo 1','Tipo 1','Tipo 2','Tipo 2','Tipo 3','Tipo 3']

<a alt="criação do dataframe">df =</a> pd.DataFrame ([x1,x2,x3,classificacao], index = ['x1', 'x2', 'x3', 'Classificação']).T

<a alt="visualização do dataframe criado">df</a>

</code></pre>
<pre><code><a alt="dataframe criado">	x1	x2	x3	Classificação</a>
0	1	7	6	Tipo 1
1	2	8	6	Tipo 1
2	1	NaN	4	Tipo 2
3	1	6	3	Tipo 2
4	3	9	9	Tipo 3
5	4	8	7	Tipo 3
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="105" name="sl">
			   <label for="105"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod996', 'cd996')" onmouseout="outFunc('cd996')"><span class="tooltiptext" id="cd996">Copiar o código</span></button></div>Eliminando uma coluna:
<pre><code id="cod996">df.drop ('x2', axis = 1)
</code></pre>
<pre><code><a alt="dataframe modificado">	x1	x3	Classificação</a>
0	1	6	Tipo 1
1	2	6	Tipo 1
2	1	4	Tipo 2
3	1	3	Tipo 2
4	3	9	Tipo 3
5	4	7	Tipo 3
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="106" name="sl">
			   <label for="106"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod995', 'cd995')" onmouseout="outFunc('cd995')"><span class="tooltiptext" id="cd995">Copiar o código</span></button></div>Selecionando somente os dados de Tipo 2:
<pre><code id="cod995">df [df ['Classificação'] == 'Tipo 2']
</code></pre>
<pre><code><a alt="dataframe modificado">	x1	x2	x3	Classificação</a>
2	1	NaN	4	Tipo 2
3	1	6	3	Tipo 2
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="107" name="sl">
			   <label for="107"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod994', 'cd994')" onmouseout="outFunc('cd994')"><span class="tooltiptext" id="cd994">Copiar o código</span></button></div>Preenchendo os valores vazios:
<pre><code id="cod994">df.fillna(value=10, inplace = True)
</code></pre>
<pre><code><a alt="dataframe modificado">	x1	x2	x3	Classificação</a>
0	1	7	6	Tipo 1
1	2	8	6	Tipo 1
2	1	<a alt="dado substituído">10</a>	4	Tipo 2
3	1	6	3	Tipo 2
4	3	9	9	Tipo 3
5	4	8	7	Tipo 3
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="108" name="sl">
			   <label for="108"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod993', 'cd993')" onmouseout="outFunc('cd993')"><span class="tooltiptext" id="cd993">Copiar o código</span></button></div>Calculando a média dos dados de cada atributo pra cada classificação:
<pre><code id="cod993">df.groupby ('Classificação').mean ()
</code></pre>
<pre><code><a alt="médias dos atributos">Classificação	x1	x2	x3</a>
Tipo 1		1.5	7.5	6.0
Tipo 2		1.0	8.0	3.5
Tipo 3		3.5	8.5	8.0
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="109" name="sl">
			   <label for="109"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod992', 'cd992')" onmouseout="outFunc('cd992')"><span class="tooltiptext" id="cd992">Copiar o código</span></button></div>Descrevendo os dados:
<pre><code id="cod992">df.describe ()
</code></pre>
<pre><code><a alt="descrição do dataframe criado">	x1		x2		x3</a>
count	6.000000	6.000000	6.000000
mean	2.000000	8.000000	5.833333
std	1.264911	1.414214	2.136976
min	1.000000 	6.000000 	3.000000
25%	1.000000 	7.250000 	4.500000
50%	1.500000 	8.000000 	6.000000
75%	2.750000 	8.750000 	6.750000
max	4.000000 	10.000000 	9.000000
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="110" name="sl">
			   <label for="110"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod991', 'cd991')" onmouseout="outFunc('cd991')"><span class="tooltiptext" id="cd991">Copiar o código</span></button></div>Somando 1 a mais nas duas primeiras colunas:
<pre><code id="cod991">df.iloc [:, 0:2] + 1
</code></pre>
<pre><code><a alt="dataframe modificado">	x1	x2</a>
0	2	8
1	3	9
2	2	11
3	2	7
4	4	10
5	5	9
</code></pre></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-49a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-50.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-51.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-52.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-53.png" loading="lazy"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo4">4. Taxonomia dos dados</summary>
  <p>Material da página 54 até a página 81.</p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-53.png" loading="lazy"/>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/zWIc2NOLsCE" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-54.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-55.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-56.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-57.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-58.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-59.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-60.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-61.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-62.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/BxU1DZDSTiY" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-63.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod31', 'cd31')" onmouseout="outFunc('cd31')"><span class="tooltiptext" id="cd31">Copiar o código</span></button></div>Gráfico radar (polar):
<pre><code id="cod31">import plotly.io as pio
pio.renderers
pio.renderers.default = 'browser'
import plotly.graph_objects as go

<a alt="rótulos das categorias de classificação">rotulos =</a> ['Composition','Vocal','Rhythm', 'Solos', 'Humour']

fig = go.Figure()

fig.add_trace(<a alt="função do gráfico radar (polar)">go.Scatterpolar</a>(r = [10, 8, 6, 5, 9], <a alt="valores de cada categoria do primeiro dado">theta =</a> rotulos, fill = 'toself', 
name = 'John', opacity = 0.3))
fig.add_trace(go.Scatterpolar(r = [10, 8, 6, 7, 6], theta = rotulos, fill = 'toself', 
name = 'Paul', opacity = 0.3))
fig.add_trace(go.Scatterpolar(r = [8, 7, 6, 10, 5], theta = rotulos, fill = 'toself', 
name = 'George', opacity = 0.3))
fig.add_trace(go.Scatterpolar(r = [2, 2, 10, 5, 5], theta = rotulos, fill = 'toself', 
name = 'Ringo', opacity = 0.3))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-63a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-64.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod32', 'cd32')" onmouseout="outFunc('cd32')"><span class="tooltiptext" id="cd32">Copiar o código</span></button></div>Gráfico com coordenadas paralelas:
<pre><code id="cod32">import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers
pio.renderers.default = 'browser'

df = pd.read_csv('C:/dados/penguin2.csv')

fig = go.Figure(data = <a alt="função do gráfico de coordenadas paralelas">go.Parcoords</a>(line = dict(color = <a alt="valores das cores para classificação">df['Cor_id']</a>,
    <a alt="escala de cores da classificação">colorscale =</a> [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']]),
    dimensions = list([dict(label = 'Compr. do bico', values = df['Comprimento do bico']),
    dict(label = 'Profundidade do bico', values = df['Profundidade do bico']),
    dict(label = 'Compr. da nadadeira', values = df['Comprimento da nadadeira']),
    dict(label = 'Massa corporal', values = df['Massa corporal']),
    dict(label = 'Espécies', values = df['Cor_id'], tickvals = [1,2,3],
    ticktext = ['Adelie', 'Gentoo', 'Chinstrap'])])))
fig.update_layout(font = dict(size = 24))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-64a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-65.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod33', 'cd33')" onmouseout="outFunc('cd33')"><span class="tooltiptext" id="cd33">Copiar o código</span></button></div>Gráfico com seleção interativa:
<pre><code id="cod33"><a alt="biblioteca bokeh de gráficos interativos">from bokeh.layouts import gridplot</a>
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap, factor_mark
import pandas as pd

df = pd.read_csv('C:/dados/penguin2.csv')
especies = ['Adelie', 'Gentoo', 'Chinstrap']
<a alt="marcador de cada espécie">markers =</a> ['hex', 'circle_x', 'triangle']

<a alt="ferramentas de interação">tools =</a> 'box_select,lasso_select,box_zoom,pan'

<a alt="separação de atributos">source =</a> ColumnDataSource(data = dict(x = df.loc[:,'Comprimento do bico'], 
	y = df.loc[:,'Profundidade do bico'], z = df.loc[:,'Comprimento da nadadeira'],
	w = df.loc[:,'Massa corporal'], esp = df.loc[:,'Espécie']))

p1 = figure(tools = tools, title = None)
p1.xaxis.axis_label = 'Comprimento do bico'
p1.yaxis.axis_label = 'Profundidade do bico'
<a alt="primeiro gráfico interativo de dispersão">p1.scatter</a>(x = 'x', y = 'y', source = source, legend_field = 'esp', fill_alpha = 0.4, size = 12,
    marker = factor_mark('esp', markers, especies),
    color = factor_cmap('esp', 'Category10_3', especies))
p1.legend.location = 'bottom_right'

p2 = figure(tools = tools, title = None)
p2.xaxis.axis_label = 'Comprimento da nadadeira'
p2.yaxis.axis_label = 'Massa corporal'
<a alt="segundo gráfico interativo de dispersão">p2.scatter</a>(x = 'z', y = 'w', source = source, legend_field = 'esp', fill_alpha = 0.4, size = 12,
    marker = factor_mark('esp', markers, especies),
    color = factor_cmap('esp', 'Category10_3', especies))
p2.legend.location = 'bottom_right'

p = gridplot([[p1, p2]])
show(p)
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-65a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-66.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-67.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod34', 'cd34')" onmouseout="outFunc('cd34')"><span class="tooltiptext" id="cd34">Copiar o código</span></button></div>Grafo orientado:
<pre><code id="cod34"><a alt="biblioteca networkx de grafos orientados">import networkx as nx</a>
import matplotlib.pyplot as plt

<a alt="definição dos arcos entre os nós">arcos =</a> [['Madrid','Paris'], ['Madrid','Bern'], ['Bern','Madrid'], ['Bern','Amsterdan'], 
    ['Bern','Berlin'], ['Bern','Rome'], ['Amsterdan','Berlin'], ['Amsterdan','Copenhagen'], 
    ['Berlin','Copenhagen'], ['Berlin','Budapest'],['Berlin','Warsaw'], ['Berlin','Rome'], 
    ['Budapest','Warsaw'], ['Budapest','Rome'], ['Budapest','Athens'], ['Budapest','Bucharest'], 
    ['Bucharest','Athens'], ['Bucharest','Ankara'], ['Bucharest','Kiev'], ['Ankara','Moscow'], 
    ['Kiev','Moscow'], ['Warsaw','Moscow'], ['Moscow','Kiev'], ['Warsaw','Kiev'], 
    ['Paris','Amsterdan'], ['Paris','Bern']]
g = <a alt="grafo orientado (direcionado)">nx.DiGraph()</a>
g.add_edges_from(arcos)
plt.figure()

<a alt="coordenadas de cada nó">pos =</a> {'Madrid': [36, 0], 'Paris': [114, 151], 'Bern': [184, 116], 'Berlin': [261, 228],
    'Amsterdan': [151, 222], 'Rome': [244, 21], 'Copenhagen': [247, 294], 'Budapest': [331, 121],
    'Warsaw': [356, 221], 'Athens': [390, -44], 'Bucharest': [422, 67], 'Ankara': [509, -13], 
    'Kiev': [480, 177], 'Moscow': [570, 300]}

<a alt="sequência das cores dos nós">cor =</a> ['orange', 'orange', 'green', 'orange', 'magenta', 'orange', 'orange', 'red', 
    'orange', 'orange', 'orange', 'red', 'orange', 'orange']

<a alt="rótulos dos valores dos arcos">rotulos =</a> {('Madrid','Paris'):'12', ('Madrid','Bern'):'15', ('Bern','Amsterdan'):'9', 
    ('Bern','Berlin'):'10', ('Bern','Rome'):'10', ('Paris','Bern'):'6', ('Amsterdan','Berlin'):'7', 
    ('Paris','Amsterdan'):'6', ('Amsterdan','Copenhagen'):'9', ('Berlin','Copenhagen'):'7',
    ('Berlin','Budapest'):'9', ('Berlin','Warsaw'):'6', ('Berlin','Rome'):'15', 
    ('Budapest','Warsaw'):'9', ('Budapest','Rome'):'12',  ('Budapest','Bucharest'):'10',
    ('Budapest','Athens'):'15', ('Bucharest','Athens'):'14',  ('Bucharest','Ankara'):'13',
    ('Ankara','Moscow'):'39', ('Bucharest','Kiev'):'12', ('Warsaw','Kiev'):'10', 
    ('Warsaw','Moscow'):'14', ('Moscow','Kiev'):'10'}

<a alt="comando para inserir os nós do grafo">nx.draw</a>(g, pos, with_labels = True, node_color = cor, edge_color = 'grey', alpha = 0.5, 
    linewidths = 1, node_size = 1250, labels = {node: node for node in g.nodes()})
<a alt="comando para inserir os rótulos dos arcos">nx.draw_networkx_edge_labels</a>(g, pos, edge_labels = rotulos, font_color = 'green')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-67a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-68.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod35', 'cd35')" onmouseout="outFunc('cd35')"><span class="tooltiptext" id="cd35">Copiar o código</span></button></div>Grafo orientado para circuito Hamiltoniano:
<pre><code id="cod35">import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

plt.figure()

<a alt="arquivo com os dados dos nós">df =</a> pd.read_csv('C:/dados/alb1000.csv')
<a alt="arquivo com a solução ótima">df_s =</a> pd.read_csv('C:/dados/alb1000_opt.csv')

g = nx.from_pandas_edgelist(df, source = 'v1', target = 'v2')
<a alt="posições dos nós com layout tipo kamada kawai">pos =</a> nx.kamada_kawai_layout(g)
nx.draw(g, pos = pos, node_color = 'grey', edge_color = 'grey', alpha = 0.1, linewidths = 0.2, 
    node_size = 40)

g1 = nx.from_pandas_edgelist(df_s, source = 'v1', target = 'v2', create_using = nx.DiGraph)
nx.draw(g1, pos = pos, node_color = 'green', edge_color = 'royalblue', alpha = 0.5,
    linewidths = 2, node_size = 40)

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo4/alb1000.csv" target="_blank">Arquivo com os dados dos nós</a></p>
  <p>&#x1f4ca; <a href="modulo4/alb1000_opt.csv" target="_blank">Arquivo com a solução ótima</a></p>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-68a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-69.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod36', 'cd36')" onmouseout="outFunc('cd36')"><span class="tooltiptext" id="cd36">Copiar o código</span></button></div>Grafo orientado para o problema do Caixeiro Viajante:
<pre><code id="cod36">import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure()

<a alt="arquivo com as posições dos nós">position =</a> np.array(pd.read_csv('C:/dados/pcb442.csv'))
<a alt="arquivo com a solução ótima">df =</a> pd.read_csv('C:/dados/pcb442_opt.csv')

i = np.arange(0, len(df))
g = nx.from_pandas_edgelist(df, source = 'v1', target = 'v2', create_using = nx.DiGraph)
pos = {}

for k in i:
    pos[k] = [position[k][1], position[k][2]]

nx.draw(g, pos = pos, node_color = 'royalblue', edge_color = 'green', alpha = 0.6, 
    linewidths = 1, node_size = 40)

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo4/pcb442.csv" target="_blank">Arquivo com os dados dos nós</a></p>
  <p>&#x1f4ca; <a href="modulo4/pcb442_opt.csv" target="_blank">Arquivo com a solução ótima</a></p>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-69a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/qoVXh9qJuOI" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-70.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-71.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod37', 'cd37')" onmouseout="outFunc('cd37')"><span class="tooltiptext" id="cd37">Copiar o código</span></button></div>Gráfico de setores:
<pre><code id="cod37">import plotly.io as pio
import plotly.express as px
pio.renderers
pio.renderers.default = 'browser'

<a alt="rótulos dos setores">setores =</a> ['Government', 'Real Estate', 'Technology, Media e Startups', 'Banking & Fiance', 
    'Economic Development', 'Health Care', 'Sports Business', 'Arts, Travel, Tourism & Ports',
    'Restaurants', 'Law', 'Transit', 'Education & Nonprofits', 'Retail & Entertainment']

<a alt="valores dos setores">valores =</a> [19, 8, 8, 8, 14, 9, 7, 6, 5, 5, 3, 5, 3]

fig = <a alt="função do gráfico de setores">px.pie</a>(values = valores, names = setores, opacity = 0.9)

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-71a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-72.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod38', 'cd38')" onmouseout="outFunc('cd38')"><span class="tooltiptext" id="cd38">Copiar o código</span></button></div>Gráfico Treeview:
<pre><code id="cod38">import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.renderers
pio.renderers.default = 'browser'

setores = ['Government', 'Real Estate', 'Technology, Media e Startups', 'Banking & Fiance', 
    'Economic Development', 'Health Care', 'Sports Business', 'Arts, Travel, Tourism & Ports',
    'Restaurants', 'Law', 'Transit', 'Education & Nonprofits', 'Retail & Entertainment']

valores = [19, 8, 8, 8, 14, 9, 7, 6, 5, 5, 3, 5, 3]

df = pd.DataFrame(dict(setores = setores, valores = valores))

df['Power 100 by Industry'] = 'Power 100 by Industry'

fig = <a alt="função do gráfico treemap">px.treemap</a>(df, path = ['Power 100 by Industry', 'setores'], values = 'valores', 
    color_continuous_scale = 'spectral', color = 'valores')

fig.update_traces(root_color = 'lightgrey', opacity = 0.9)
fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-72a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-73.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod39', 'cd39')" onmouseout="outFunc('cd39')"><span class="tooltiptext" id="cd39">Copiar o código</span></button></div>Gráfico Treeview:
<pre><code id="cod39">import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.renderers
pio.renderers.default = 'browser'

setores = ['Government', 'Real Estate', 'Technology, Media e Startups', 'Banking & Fiance', 
    'Economic Development', 'Health Care', 'Sports Business', 'Arts, Travel, Tourism & Ports',
    'Restaurants', 'Law', 'Transit', 'Education & Nonprofits', 'Retail & Entertainment']

valores = [19, 8, 8, 8, 14, 9, 7, 6, 5, 5, 3, 5, 3]

<a alt="separação em categorias">categorias =</a> ['State', 'State', 'Technology', 'State', 'Technology', 'State', 'Entertainment', 
    'Entertainment', 'Entertainment', 'State', 'State', 'State', 'Entertainment']

df = pd.DataFrame(dict(setores = setores, valores = valores, categorias = categorias))

df['Power 100 by Industry'] = 'Power 100 by Industry'

fig = px.treemap(df, path = ['Power 100 by Industry', 'categorias', 'setores'], 
	values = 'valores', color_continuous_scale = 'spectral', color = 'valores')

fig.update_traces(root_color = 'lightgrey', opacity = 0.9)
fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-73a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod40', 'cd40')" onmouseout="outFunc('cd40')"><span class="tooltiptext" id="cd40">Copiar o código</span></button></div>Gráfico Treeview:
<pre><code id="cod40">import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
pio.renderers
pio.renderers.default = 'browser'

df = pd.read_csv('C:/dados/treemap1.csv')

fig = px.treemap(df, path = [px.Constant('IEEE Spectrum'), 'Country', 'Company'], 
    values = 'Sales 2005', <a alt="separação em categorias">color =</a> 'Sales 2005', hover_data = ['Sales 2006'],
    <a alt="mapa de cores">color_continuous_scale =</a> 'rainbow', 
    color_continuous_midpoint = np.average(df['Sales 2006'], weights = df['Rank 2006']))

fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo4/treemap1.csv" target="_blank">Arquivo de dados para o Treemap</a></p>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod41', 'cd41')" onmouseout="outFunc('cd41')"><span class="tooltiptext" id="cd41">Copiar o código</span></button></div>Gráfico Sunburst (aneis aninhados):
<pre><code id="cod41">import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.renderers
pio.renderers.default = 'browser'

setores = ['Government', 'Real Estate', 'Technology, Media e Startups', 'Banking & Fiance', 
    'Economic Development', 'Health Care', 'Sports Business', 'Arts, Travel, Tourism & Ports',
    'Restaurants', 'Law', 'Transit', 'Education & Nonprofits', 'Retail & Entertainment']

valores = [19, 8, 8, 8, 14, 9, 7, 6, 5, 5, 3, 5, 3]

categorias = ['State', 'State', 'Technology', 'State', 'Technology', 'State', 'Entertainment', 
    'Entertainment', 'Entertainment', 'State', 'State', 'State', 'Entertainment']

df = pd.DataFrame(dict(setores = setores, valores = valores, categorias = categorias))

df['Power 100 by Industry'] = 'Power 100 by Industry'

fig = <a alt="função para criar o gráfico Sunburst">px.sunburst</a>(df, path = ['Power 100 by Industry', 'setores'], values = 'valores', 
    color_continuous_scale = 'spectral', color = 'valores')

fig.update_traces(root_color = 'lightgrey', opacity = 0.9)
fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74b.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod42', 'cd42')" onmouseout="outFunc('cd42')"><span class="tooltiptext" id="cd42">Copiar o código</span></button></div>Gráfico Sunburst (aneis aninhados):
<pre><code id="cod42">import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.renderers
pio.renderers.default = 'browser'

setores = ['Government', 'Real Estate', 'Technology, Media e Startups', 'Banking & Fiance', 
    'Economic Development', 'Health Care', 'Sports Business', 'Arts, Travel, Tourism & Ports',
    'Restaurants', 'Law', 'Transit', 'Education & Nonprofits', 'Retail & Entertainment']

valores = [19, 8, 8, 8, 14, 9, 7, 6, 5, 5, 3, 5, 3]

categorias = ['State', 'State', 'Technology', 'State', 'Technology', 'State', 'Entertainment', 
    'Entertainment', 'Entertainment', 'State', 'State', 'State', 'Entertainment']

df = pd.DataFrame(dict(setores = setores, valores = valores, categorias = categorias))

df['Power 100 by Industry'] = 'Power 100 by Industry'

fig = px.sunburst(df, <a alt="separação por categorias">path =</a> ['Power 100 by Industry', 'categorias', 'setores'], 
	values = 'valores', color_continuous_scale = 'spectral', color = 'valores')

fig.update_traces(root_color = 'lightgrey', opacity = 0.9)
fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74c.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod43', 'cd43')" onmouseout="outFunc('cd43')"><span class="tooltiptext" id="cd43">Copiar o código</span></button></div>Gráfico Sunburst (aneis aninhados):
<pre><code id="cod43">import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
pio.renderers
pio.renderers.default = 'browser'

df = pd.read_csv('C:/dados/treemap1.csv')

fig = <a alt="função para criar o gráfico Sunburst">px.sunburst</a>(df, path = [px.Constant('IEEE Spectrum'), 'Country', 'Company'], 
    values = 'Sales 2005', color = 'Sales 2005', hover_data = ['Sales 2006'],
    color_continuous_scale = 'rainbow', 
    color_continuous_midpoint = np.average(df['Sales 2006'], weights = df['Rank 2006']))

fig.update_layout(margin = dict(t = 25, l = 25, r = 25, b = 25))

fig.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-75.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-76.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod44', 'cd44')" onmouseout="outFunc('cd44')"><span class="tooltiptext" id="cd44">Copiar o código</span></button></div>Gráfico RadViz:
<pre><code id="cod44">import pandas as pd
from matplotlib import pyplot as plt

pinguin = pd.read_csv('C:/dados/penguin2.csv', header = 0, <a alt="colunas que contém os dados usados">usecols =</a> [1,3,4,5,6,8])

ax = plt.grid(color = '#d5f8e3', linewidth = 0.5)
fig = <a alt="função para criar o gráfico RadViz">pd.plotting.radviz</a>(pinguin, <a alt="critério de separação dos dados">'Espécie'</a>, colormap = 'rainbow', alpha = 0.6, ax = ax)

fig.show
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod45', 'cd45')" onmouseout="outFunc('cd45')"><span class="tooltiptext" id="cd45">Copiar o código</span></button></div>Gráfico de enxame (swarm):
<pre><code id="cod45">import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico swarm">sns.swarmplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', <a alt="critério de separação">hue = 'Sexo'</a>, data = pinguin)

</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77b.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod46', 'cd46')" onmouseout="outFunc('cd46')"><span class="tooltiptext" id="cd46">Copiar o código</span></button></div>Gráfico de enxame (swarm) com diagrama em caixas (boxplot):
<pre><code id="cod46">import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico boxplot">sns.boxplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', data = pinguin)
sns.swarmplot(x = 'Comprimento da nadadeira', y = 'Espécie', hue = 'Sexo', data = pinguin)
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-78.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod47', 'cd47')" onmouseout="outFunc('cd47')"><span class="tooltiptext" id="cd47">Copiar o código</span></button></div>Gráfico de enxame (swarm) com violino:
<pre><code id="cod47">import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico violino">sns.violinplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', data = pinguin, 
    palette = 'Oranges')
sns.swarmplot(x = 'Comprimento da nadadeira', y = 'Espécie', hue = 'Sexo', data = pinguin,  
    palette = 'Blues')
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-78a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-79.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod48', 'cd48')" onmouseout="outFunc('cd48')"><span class="tooltiptext" id="cd48">Copiar o código</span></button></div>Reconhecimento de imagens:
<pre><code id="cod48"><a alt="biblioteca para carregar imagens">from PIL import Image</a>
import numpy as np
import matplotlib.pyplot as plt

<a alt="função para converter os pixels da imagem em RGB">im =</a> Image.open('C:/dados/imagem.png').convert('RGB')
          
<a alt="função para transformar a imagem em um array">imf =</a> np.asarray(im)
<a alt="função para desenhar a imagem">imf.tofile</a>('C:/dados/testeImagem.csv', sep = ',')

print(im.size)
plt.imshow(imf)
print(imf)
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo4/imagem.png" target="_blank">Imagem de 16x16 pixels</a></p>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-79a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-80.png" loading="lazy"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo5">5. Linhas, polígonos, poliedros e superfícies</summary>
  <p>Material da página 81 até a página 92.</p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-80.png" loading="lazy"/>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/CV7nSgcqYpU" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod49', 'cd49')" onmouseout="outFunc('cd49')"><span class="tooltiptext" id="cd49">Copiar o código</span></button></div>Construção de um retângulo:
<pre><code id="cod49">import matplotlib.pyplot as plt
<a alt="bibliotecas de polígonos e curvas">from matplotlib.path import Path
import matplotlib.patches as patches</a>

<a alt="sequência de vértices">verts =</a> [(0, 0), (0, 1), (2, 1), (2, 0), (0, 0),]
<a alt="construção do caminho de vértices">path =</a> Path(verts)

fig, ax = plt.subplots()
patch = <a alt="função para desenhar o caminho path">patches.PathPatch</a>(path, facecolor = 'aqua', linestyle = '--', linewidth = 2, 
    edgecolor = 'red')
ax.add_patch(patch)

ax.set_xlim(-1, 2.5)
ax.set_ylim(-1, 1.5)
<a alt="ajuste da escala dos eixos">plt.gca()</a>.set_aspect('equal', adjustable = 'box')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod50', 'cd50')" onmouseout="outFunc('cd50')"><span class="tooltiptext" id="cd50">Copiar o código</span></button></div>Construção de uma elipse:
<pre><code id="cod50">import matplotlib.pyplot as plt
<a alt="biblioteca de elipses">from matplotlib.patches import Ellipse</a>

fig, ax = plt.subplots()
patch = <a alt="função para desenhar a elipse">Ellipse</a>((0.5, 0.5), <a alt="diâmetro maior">0.7</a>, <a alt="diâmetro menor">0.3</a>, color = 'orange')
ax.add_patch(patch)

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod51', 'cd51')" onmouseout="outFunc('cd51')"><span class="tooltiptext" id="cd51">Copiar o código</span></button></div>Construção da superfície lateral de um cilindro circular reto:
<pre><code id="cod51">import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

raio = 3
altura = 7
x = np.linspace(-raio, raio, 100)
z = np.linspace(0, altura, 100)
<a alt="grade de x e z com limites do raio e da altura">x1, z1 =</a> np.meshgrid(x, z)
<a alt="relação que define os círculos das bases do cilindro">y1 =</a> np.sqrt(raio**2-x1**2)

rstride = 10
cstride = 10
<a alt="função para desenhar a superfície lateral">ax.plot_surface</a>(x1, y1, z1, alpha = 0.7, color = 'green', rstride = rstride, cstride = cstride)
ax.plot_surface(x1, -y1, z1, alpha = 0.7, color = 'green', rstride = rstride, cstride= cstride)

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod52', 'cd52')" onmouseout="outFunc('cd52')"><span class="tooltiptext" id="cd52">Copiar o código</span></button></div>Construção de um cilindro circular reto:
<pre><code id="cod52">import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

raio = 3
altura = 7
x = np.linspace(-raio, raio, 100)
z = np.linspace(0, altura, 100)
x1, z1 = np.meshgrid(x, z)
y1 = np.sqrt(raio**2-x1**2)

rstride = 10
cstride = 10
ax.plot_surface(x1, y1, z1, alpha = 0.7, color = 'green', rstride = rstride, cstride = cstride)
ax.plot_surface(x1, -y1, z1, alpha = 0.7, color = 'green', rstride = rstride, cstride= cstride)

<a alt="bibliotecas para desenhar o círculo em 3D">from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d</a>

p = <a alt="função para desenhar o círculo">Circle</a>(<a alt="coordenadas do centro">(0, 0)</a>, raio, color = 'red', alpha = 0.5)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z = 0, zdir = 'z')
p = Circle((0, 0), raio, color = 'red', alpha = 0.5)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z = altura, zdir = 'z')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-83.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod53', 'cd53')" onmouseout="outFunc('cd53')"><span class="tooltiptext" id="cd53">Copiar o código</span></button></div>Construção de um cone circular reto:
<pre><code id="cod53">import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

raio = 3
altura = 7
x = np.linspace(-raio, raio, 155)
z = np.linspace(0, altura, 155)
<a alt="grade de x e z limitada ao raio e a altura do cone">x1, z1 = </a>np.meshgrid(x, z)
<a alt="relação entre as coordenadas para definir a superfície do cone">y1 =</a> np.sqrt(z1**2*(raio/altura)**2 - x1**2)

rstride = 10
cstride = 10
ax.plot_surface(x1, y1, -z1, alpha = 0.7, color = 'grey', rstride = rstride, cstride = cstride)
ax.plot_surface(x1, -y1, -z1, alpha = 0.7, color = 'grey', rstride = rstride, cstride= cstride)

<a alt="base do cone">p = Circle</a>((0, 0), raio, color = 'aqua', alpha = 0.5)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z = -altura, zdir = 'z')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-83a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod54', 'cd54')" onmouseout="outFunc('cd54')"><span class="tooltiptext" id="cd54">Copiar o código</span></button></div>Construção de um cone circular reto (coordenadas polares):
<pre><code id="cod54">import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

raio = 4
altura = 7
altura1 = np.linspace(0, altura, 150)
raio1 = np.linspace(0, raio, 150)
theta = np.linspace(0, 2*np.pi, 150)

R, T = np.meshgrid(raio1, theta)
A, T = np.meshgrid(altura1, theta)
<a alt="coordenadas polares paramétricas">X, Y, Z =</a> R*np.cos(T), R*np.sin(T), A

p = Circle((0, 0), raio, color = 'aqua', alpha = 0.2)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z = -altura, zdir = 'z')

rstride = 10
cstride = 10
ax.plot_surface(X, Y, -Z, alpha = 0.7, color = 'grey', rstride = rstride, cstride = cstride)

plt.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-84.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-85.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod55', 'cd55')" onmouseout="outFunc('cd55')"><span class="tooltiptext" id="cd55">Copiar o código</span></button></div>Construção de uma superfície:
<pre><code id="cod55">import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

r = 3
x = np.linspace(-r, r, 155)
y = np.linspace(-r, r, 155)
x1, y1 = np.meshgrid(x, y)
<a alt="função de x e y">z1 =</a> x1*np.exp(-x1**2 - y1**4)

ax.plot_surface(x1, y1, z1, alpha = 0.7, <a alt="mapa de cores cool invertido">cmap =</a> 'cool_r')

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-85a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod56', 'cd56')" onmouseout="outFunc('cd56')"><span class="tooltiptext" id="cd56">Copiar o código</span></button></div>Construção de um poliedro:
<pre><code id="cod56">from matplotlib import pyplot as plt
<a alt="biblioteca para construção de poliedros">from mpl_toolkits.mplot3d.art3d import Poly3DCollection</a>
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

<a alt="coordenadas dos vértices">v =</a> np.array([[1, 1, 0], [5, 1, 0], [5, 5, 0], [1, 5, 0], [1, 1, 4], [5, 1, 4], [5, 5, 4], 
    [1, 5, 4]])

<a alt="definição do conjunto de vértices de cada face">faces =</a> [[v[0],v[1],v[2],v[3]], [v[0],v[1],v[5],v[4]], [v[0],v[3],v[7],v[4]],
    [v[3],v[2],v[6],v[7]], [v[1],v[2],v[6],v[5]], [v[4],v[5],v[6],v[7]]]

<a alt="gráfico de dispersão dos vértices">ax.scatter3D</a>(v[:, 0], v[:, 1], v[:, 2])

<a alt="construção do poliedro com o conjunto definido de faces">ax.add_collection3d</a>(Poly3DCollection(faces, facecolors = 'orange', edgecolors = 'blue', 
    alpha = 0.25))

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod57', 'cd57')" onmouseout="outFunc('cd57')"><span class="tooltiptext" id="cd57">Copiar o código</span></button></div>Construção de um poliedro com rótulos dos vértices:
<pre><code id="cod57">from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

v = np.array([[1, 1, 0], [5, 1, 0], [5, 5, 0], [1, 5, 0], [1, 1, 4], [5, 1, 4], [5, 5, 4], 
[1, 5, 4]])

faces = [[v[0],v[1],v[2],v[3]], [v[0],v[1],v[5],v[4]], [v[0],v[3],v[7],v[4]],
         [v[3],v[2],v[6],v[7]], [v[1],v[2],v[6],v[5]], [v[4],v[5],v[6],v[7]]]

ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

<a alt="conjunto de cores das faces">cores =</a> ['blue', 'green', 'yellow', 'red', 'cyan', 'black']

ax.add_collection3d(Poly3DCollection(faces, <a alt="cores das faces">facecolors =</a> cores, edgecolors = 'blue', 
    alpha = 0.25))

x = v[:, 0]
y = v[:, 1]
z = v[:, 2]
<a alt="rótulos dos vértices">rotulos =</a> ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

<a alt="laço para inserir os respectivos rótulos dos vértices">for x, y, z, tag in zip(x, y, z, rotulos):</a>
    label = tag
    ax.text3D(x, y, z, label, zdir = [1,1,1], color = 'k')

<a alt="mesma escala usada nos três eixos">ax.set_box_aspect((np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2])))</a>

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <div class="embed-container">
		<iframe width="100%" src="https://www.youtube.com/embed/UZnpu-CP0tI" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-87.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod58', 'cd58')" onmouseout="outFunc('cd58')"><span class="tooltiptext" id="cd58">Copiar o código</span></button></div>Construção de uma superfície com triangulação:
<pre><code id="cod58">import matplotlib.pyplot as plt
import numpy as np

n_raio = 10
n_angulos = 48
raio = np.linspace(0.125, 1.0, n_raio)
angulo = np.linspace(0, 2*np.pi, n_angulos, endpoint = False)[..., np.newaxis]

x = np.append(0, (raio*np.cos(angulo)).flatten())
y = np.append(0, (raio*np.sin(angulo)).flatten())
z = np.sin(-x*y)

ax = plt.figure().add_subplot(projection = '3d')
<a alt="triangulação da superfície">ax.plot_trisurf</a>(x, y, z, linewidth = 0.2, cmap = 'RdBu')

ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-87a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod59', 'cd59')" onmouseout="outFunc('cd59')"><span class="tooltiptext" id="cd59">Copiar o código</span></button></div>Construção de uma superfície com coordenadas de um arquivo:
<pre><code id="cod59">import numpy as np
import matplotlib.pyplot as plt

vertices = <a alt="coordenadas dos pontos da superfície">np.array(np.loadtxt</a>('C:/dados/volcano.txt', int))
x = vertices[:,0]
y = vertices[:,1]
z = vertices[:,2]

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

ax.plot_trisurf(x, y, z, cmap = 'jet_r', edgecolor = 'grey', linewidth = 0.15, alpha = 0.7)

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo5/volcano.txt" target="_blank">Arquivo com as coordenadas da superfície</a></p>
  </details></div>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-88.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://python-graph-gallery.com/371-surface-plot" target="_blank">https://python-graph-gallery.com/371-surface-plot</a></p></details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-88a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-89.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod60', 'cd60')" onmouseout="outFunc('cd60')"><span class="tooltiptext" id="cd60">Copiar o código</span></button></div>Triangulação de um objeto 3D de extensão PLY:
<pre><code id="cod60"><a alt="biblioteca para leitura de arquivo PLY">from plyfile import PlyData</a>
import numpy as np
import matplotlib.pyplot as plt

plydata = <a alt="leitura do arquivo PLY">PlyData.read</a>('C:/dados/galleon.ply')
with open('C:/dados/galleon.ply', 'rb') as f:
    plydata = PlyData.read(f)

plydata.elements[0].name
plydata.elements[0].data[0]
nr_vertices = plydata.elements[0].count
nr_faces = plydata.elements[1].count

<a alt="extração das informações dos vértices do objeto">vertices =</a> np.array([plydata['vertex'][k] for k in range(nr_vertices)])
x, y, z = zip(*vertices)

<a alt="extração das informações das faces do objeto">faces =</a> [plydata['face'][k][0] for k in range(nr_faces)]
ax = plt.figure().add_subplot(projection = '3d')

ax.plot_trisurf(x, y, z, triangles = faces, cmap = 'RdBu_r', edgecolor = 'green', 
    linewidth = 0.1, alpha = 0.5)

<a alt="escala igual para os eixos">ax.set_box_aspect</a>((np.ptp(x), np.ptp(y), np.ptp(z)))

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo5/galleon.ply" target="_blank">Arquivo galleon PLY</a></p>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-89a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod61', 'cd61')" onmouseout="outFunc('cd61')"><span class="tooltiptext" id="cd61">Copiar o código</span></button></div>Triangulação de um objeto 3D de extensão PLY:
<pre><code id="cod61">from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt

plydata = PlyData.read('C:/dados/chopper.ply')
with open('C:/dados/chopper.ply', 'rb') as f:
    plydata = PlyData.read(f)

plydata.elements[0].name
plydata.elements[0].data[0]
nr_vertices = plydata.elements[0].count
nr_faces = plydata.elements[1].count

vertices = np.array([plydata['vertex'][k] for k in range(nr_vertices)])
x, y, z = zip(*vertices)

faces = [plydata['face'][k][0] for k in range(nr_faces)]
ax = plt.figure().add_subplot(projection = '3d')

ax.plot_trisurf(x, y, z, triangles = faces, cmap = 'RdBu_r', edgecolor = 'green', 
linewidth = 0.1, alpha = 0.5)

ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

<a alt="eixos e planos de projeções ocultos">plt.axis('off')</a>

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo5/chopper.ply" target="_blank">Arquivo chopper PLY</a></p>
  </details></div>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-90.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html" target="_blank">https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html</a></p></details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-90a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-91.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod62', 'cd62')" onmouseout="outFunc('cd62')"><span class="tooltiptext" id="cd62">Copiar o código</span></button></div>Triangulação de um objeto 3D:
<pre><code id="cod62">import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

<a alt="arquivo dos vértices">vertices =</a> np.loadtxt('C:/dados/vertices_hind.txt')
<a alt="arquivo das faces">faces =</a> np.loadtxt('C:/dados/faces_hind.txt', int)

facesc = np.array(vertices[faces])

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

ax.add_collection3d(Poly3DCollection(facesc, facecolors = 'green', edgecolors = 'grey', 
    alpha = 0.25, linewidth = 0.1))

ax.set_xlim3d(np.min(vertices[:,0]), np.max(vertices[:,0]))
ax.set_ylim3d(np.min(vertices[:,1]), np.max(vertices[:,1]))
ax.set_zlim3d(np.min(vertices[:,2]), np.max(vertices[:,2]))

ax.set_box_aspect((np.ptp(vertices[:,0]), np.ptp(vertices[:,1]), np.ptp(vertices[:,2])))

plt.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo5/vertices_hind.txt" target="_blank">Arquivo dos vértices</a></p>
  <p>&#x1f4ca; <a href="modulo5/faces_hind.txt" target="_blank">Arquivo das faces</a></p>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-91a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo6">6. Modelos de iluminação</summary>
  <p>Material da página 92 até a página 105.</p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-91.png" loading="lazy"/>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/8qseMynFIcg" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-92.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod63', 'cd63')" onmouseout="outFunc('cd63')"><span class="tooltiptext" id="cd63">Copiar o código</span></button></div>Cena com eixos e um cilindro programados com VTK:
<pre><code id="cod63"><a alt="Conexões com as bibliotecas que serão usadas para renderizar os atores">import vtkmodules.vtkRenderingOpenGL2</a>
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def main():
    <a alt="Dados de entrada: cor do fundo e cilindro">colors =</a> vtkNamedColors()
    colors.SetColor("BkgColor", [0.95, 0.95, 1, 0])
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(30)

    <a alt="Mapeamento da geometria do cilindro">cylinderMapper = </a>vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    <a alt="Criação do ator">cylinderActor =</a> vtkActor()
    <a alt="Adiciona as propriedades primitivas do cilindro">cylinderActor.SetMapper(cylinderMapper)</a>
    cylinderActor.GetProperty().SetColor(colors.GetColor3d("Yellow"))
    <a alt="Propriedades do cilindro (ator)">cylinder.SetRadius(0.5)</a>
    cylinder.SetHeight(1.5)
    cylinderActor.SetPosition(2,-1,1.5)
    cylinderActor.RotateZ(-30.0)
    cylinderActor.RotateX(-30.0)
    
    <a alt="Renderização (janela e interação com o usuário)">ren = </a>vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    <a alt="Adição do cilindro e de mais um ator: eixos">ren.AddActor(cylinderActor)</a>
    axes = vtkAxesActor()
    ren.AddActor(axes)
    ren.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(500, 500)

    <a alt="Iniciar a câmera: zoom e renderização da cena">iren.Initialize()</a>
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.2)
    renWin.Render()
    iren.Start()
if __name__ == '__main__':
    main()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-93.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Etapas da renderização de uma cena com VTK</summary>
	<p>Vamos acompanhar o esquema com as etapas da criação de uma cena usando a biblioteca VTK - Visualization Toolkit.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="219" name="sl">
			   <label for="219"></label>
			   <img src="modulo6/1.png" loading="lazy"/>
			   <figcaption>Depois de criarmos as ligações com as bibliotecas do VTK, podemos definir quais serão os atores da cena (polígonos, objetos 3D, poliedros e eixos).</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="220" name="sl">
			   <label for="220"></label>
			   <img src="modulo6/2.png" loading="lazy"/>
			   <figcaption>Com os atores da cena definidos, utilizamos as propriedades para cada ator (cores, texturas, tamanhos e posições).</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="221" name="sl">
			   <label for="221"></label>
			   <img src="modulo6/3.png" loading="lazy"/>
			   <figcaption>A renderização da cena pode ser definida com a inicialização da câmera.</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="222" name="sl">
			   <label for="222"></label>
			   <img src="modulo6/4.png" loading="lazy"/>
			   <figcaption>Na etapa seguinte, definimos a iluminação da cena (posição, tipo de iluminação e cor).</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="223" name="sl">
			   <label for="223"></label>
			   <img src="modulo6/5.png" loading="lazy"/>
			   <figcaption>A janela de visualização deve ser definida, onde serão mostrados os elementos programados da cena.</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="224" name="sl">
			   <label for="224"></label>
			   <img src="modulo6/6.png" loading="lazy"/>
			   <figcaption>Para finalizar, podemos indicar quais serão os tipos de interação usados pelo usuário com os atores da cena.</figcaption>
		   </li>
		</ul>
		<img src="modulo6/1.png" class="fundo" loading="lazy"/>
  </details></div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-93a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-94.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-95.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod64', 'cd64')" onmouseout="outFunc('cd64')"><span class="tooltiptext" id="cd64">Copiar o código</span></button></div>Variação de iluminação ambiente com VTK:
<pre><code id="cod64">import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkLight,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def main():
    <a alt="Dados de entrada: cor do fundo, cilindros e esferas">colors = </a>vtkNamedColors()
    colors.SetColor('bkg', [0.65, 0.75, 0.99, 0])
    sphere = vtkSphereSource()
    sphere.SetThetaResolution(100)
    sphere.SetPhiResolution(50)
    sphere.SetRadius(0.3)
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(30)
    cylinder.SetRadius(0.3)
    cylinder.SetHeight(0.7)

    <a alt="Mapeamentos">sphereMapper =</a> vtkPolyDataMapper()
    cylinderMapper = vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphere.GetOutputPort())
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    quantidade = 8
    spheres = list()
    cylinders = list()
    <a alt="Neste exemplo, vamos visualizar apenas a luz ambiente">ambient =</a> 0.125
    diffuse = 0.0
    specular = 0.0
    position = [1.5, 0, 0]
    position1 = [2, 0, -0.5]
    <a alt="Loop para criar os Atores e definir iluminação de cada par">for i in range(0, quantidade):</a>
        spheres.append(vtkActor())
        spheres[i].SetMapper(sphereMapper)
        spheres[i].GetProperty().SetColor(colors.GetColor3d('Red'))
        spheres[i].GetProperty().SetAmbient(ambient)
        spheres[i].GetProperty().SetDiffuse(diffuse)
        spheres[i].GetProperty().SetSpecular(specular)
        spheres[i].AddPosition(position)
        cylinders.append(vtkActor())
        cylinders[i].SetMapper(cylinderMapper)
        cylinders[i].GetProperty().SetColor(colors.GetColor3d('Blue'))
        cylinders[i].GetProperty().SetAmbient(ambient)
        cylinders[i].GetProperty().SetDiffuse(diffuse)
        cylinders[i].GetProperty().SetSpecular(specular)
        cylinders[i].AddPosition(position1)
        <a alt="Incremento no valor da incidência da luz">ambient +=</a> 0.125
        position[0] += 1.25
        position1[0] += 1.25
        if i == 3:
            position[0] = 1.5
            position[1] = -1
            position1[0] = 2
            position1[1] = -1

    <a alt="Renderização">ren =</a> vtkRenderer()
    axes = vtkAxesActor()
    ren.AddActor(axes)
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for i in range(0, quantidade):
        ren.AddActor(spheres[i])
        ren.AddActor(cylinders[i])

    ren.SetBackground(colors.GetColor3d('bkg'))
    renWin.SetSize(940, 480)
    renWin.SetWindowName('AmbientSpheres')

    light = vtkLight()
    light.SetFocalPoint(1.8, 0.6, 0)
    <a alt="Posição da fonte de luz">light.SetPosition(0.8, 1.6, 1)</a>
    ren.AddLight(light)

    iren.Initialize()
    renWin.Render()
    iren.Start()
if __name__ == '__main__':
    main()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-96.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod65', 'cd65')" onmouseout="outFunc('cd65')"><span class="tooltiptext" id="cd65">Copiar o código</span></button></div>Variação de iluminação specular com VTK:
<pre><code id="cod65">import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkLight,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def main():
    colors = vtkNamedColors()
    colors.SetColor('bkg', [0.65, 0.75, 0.99, 0])
    sphere = vtkSphereSource()
    sphere.SetThetaResolution(100)
    sphere.SetPhiResolution(50)
    sphere.SetRadius(0.3)
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(30)
    cylinder.SetRadius(0.3)
    cylinder.SetHeight(0.7)

    sphereMapper = vtkPolyDataMapper()
    cylinderMapper = vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphere.GetOutputPort())
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    quantidade = 8
    spheres = list()
    cylinders = list()
    ambient = 0.8
    diffuse = 0.0
    <a alt="Neste exemplo, vamos variar apenas a luz specular">specular =</a> 0.125
    <a alt="Expoente da luz specular">specularPower =</a> 1
    position = [1.5, 0, 0]
    position1 = [2, 0, -0.5]
    for i in range(0, quantidade):
        spheres.append(vtkActor())
        spheres[i].SetMapper(sphereMapper)
        spheres[i].GetProperty().SetColor(colors.GetColor3d('Red'))
        spheres[i].GetProperty().SetAmbient(ambient)
        spheres[i].GetProperty().SetDiffuse(diffuse)
        spheres[i].GetProperty().SetSpecular(specular)
        spheres[i].GetProperty().SetSpecularPower(specularPower)
        spheres[i].GetProperty().SetSpecularColor(colors.GetColor3d('White'))
        spheres[i].AddPosition(position)
        cylinders.append(vtkActor())
        cylinders[i].SetMapper(cylinderMapper)
        cylinders[i].GetProperty().SetColor(colors.GetColor3d('Blue'))
        cylinders[i].GetProperty().SetAmbient(ambient)
        cylinders[i].GetProperty().SetDiffuse(diffuse)
        cylinders[i].GetProperty().SetSpecular(specular)
        cylinders[i].GetProperty().SetSpecularPower(specularPower)
        cylinders[i].GetProperty().SetSpecularColor(colors.GetColor3d('White'))
        cylinders[i].AddPosition(position1)
        <a alt="Incremento no valor da incidência da luz">specular +=</a> 0.125
        specularPower += 0.5
        position[0] += 1.25
        position1[0] += 1.25
        if i == 3:
            position[0] = 1.5
            position[1] = -1
            position1[0] = 2
            position1[1] = -1

    ren = vtkRenderer()
    axes = vtkAxesActor()
    ren.AddActor(axes)
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for i in range(0, quantidade):
        ren.AddActor(spheres[i])
        ren.AddActor(cylinders[i])

    ren.SetBackground(colors.GetColor3d('bkg'))
    renWin.SetSize(940, 480)
    renWin.SetWindowName('AmbientSpheres')

    light = vtkLight()
    light.SetFocalPoint(1.8, 0.6, 0)
    light.SetPosition(0.8, 1.6, 1)
    ren.AddLight(light)

    iren.Initialize()
    renWin.Render()
    iren.Start()
if __name__ == '__main__':
    main()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-97.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod66', 'cd66')" onmouseout="outFunc('cd66')"><span class="tooltiptext" id="cd66">Copiar o código</span></button></div>Iluminação de 2 fontes de luz com VTK:
<pre><code id="cod66">import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
<a alt="biblioteca de texto 3D">from vtkmodules.vtkRenderingFreeType import vtkVectorText</a>
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkLight,
    vtkPolyDataMapper,
    vtkRenderWindow,
    <a alt="biblioteca de texto 3D">vtkFollower,</a>
    vtkRenderWindowInteractor,
    vtkRenderer
)

def main():
    colors = vtkNamedColors()
    colors.SetColor('bkg', [0.65, 0.75, 0.99, 0])
    sphere = vtkSphereSource()
    sphere.SetThetaResolution(100)
    sphere.SetPhiResolution(50)
    sphere.SetRadius(0.3)
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(30)
    cylinder.SetRadius(0.3)
    cylinder.SetHeight(0.7)

    sphereMapper = vtkPolyDataMapper()
    cylinderMapper = vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphere.GetOutputPort())
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    spheres = list()
    cylinders = list()
    ambient = 0.8
    diffuse = 0.0
    specular = 0.75
    specularPower = 4
    position = [1.5, 0, 0]
    position1 = [2, 0, -0.5]
    spheres.append(vtkActor())
    spheres[0].SetMapper(sphereMapper)
    spheres[0].GetProperty().SetColor(colors.GetColor3d('Red'))
    spheres[0].GetProperty().SetAmbient(ambient)
    spheres[0].GetProperty().SetDiffuse(diffuse)
    spheres[0].GetProperty().SetSpecular(specular)
    spheres[0].GetProperty().SetSpecularPower(specularPower)
    spheres[0].GetProperty().SetSpecularColor(colors.GetColor3d('White'))
    spheres[0].AddPosition(position)
    cylinders.append(vtkActor())
    cylinders[0].SetMapper(cylinderMapper)
    cylinders[0].GetProperty().SetColor(colors.GetColor3d('Blue'))
    cylinders[0].GetProperty().SetAmbient(ambient)
    cylinders[0].GetProperty().SetDiffuse(diffuse)
    cylinders[0].GetProperty().SetSpecular(specular)
    cylinders[0].GetProperty().SetSpecularPower(specularPower)
    cylinders[0].GetProperty().SetSpecularColor(colors.GetColor3d('White'))
    cylinders[0].AddPosition(position1)
    <a alt="texto da fonte de luz 1">atext =</a> vtkVectorText()
    atext.SetText('Fonte 1')
    textMapper = vtkPolyDataMapper()
    textMapper.SetInputConnection(atext.GetOutputPort())
    textActor = vtkFollower()
    textActor.SetMapper(textMapper)
    textActor.SetScale(0.2, 0.2, 0.2)
    textActor.AddPosition(-3, 2, 0)
    <a alt="texto da fonte de luz 2">atext1 =</a> vtkVectorText()
    atext1.SetText('Fonte 2')
    textMapper = vtkPolyDataMapper()
    textMapper.SetInputConnection(atext1.GetOutputPort())
    textActor1 = vtkFollower()
    textActor1.SetMapper(textMapper)
    textActor1.SetScale(0.2, 0.2, 0.2)
    textActor1.AddPosition(3, 2, 0)

    ren = vtkRenderer()
    axes = vtkAxesActor()
    ren.AddActor(axes)
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    ren.AddActor(textActor)
    ren.AddActor(textActor1)
    ren.AddActor(spheres[0])
    ren.AddActor(cylinders[0])

    ren.SetBackground(colors.GetColor3d('bkg'))
    renWin.SetSize(940, 480)
    renWin.SetWindowName('AmbientSpheres')

    <a alt="posição da fonte de luz 1">light =</a> vtkLight()
    light.SetFocalPoint(1.8, 0.6, 0)
    light.SetPosition(-3, 2, 0)
    ren.AddLight(light)
        
    <a alt="posição da fonte de luz 2">light1 =</a> vtkLight()
    light1.SetFocalPoint(1.8, 0.6, 0)
    light1.SetPosition(3, 2, 0)
    ren.AddLight(light1)

    iren.Initialize()
    renWin.Render()
    iren.Start()
if __name__ == '__main__':
    main()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/4n8xrXzlMU4" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-98.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod67', 'cd67')" onmouseout="outFunc('cd67')"><span class="tooltiptext" id="cd67">Copiar o código</span></button></div>Criação de uma cena com Pyvista:
<pre><code id="cod67"><a alt="biblioteca Pyvista">import pyvista</a>
import pyvista as pv

filename = 'C:/dados/chopper.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()

p = <a alt="comando para renderizar o objeto 3D">pv.Plotter</a>(lighting = 'none', window_size = [1000, 1000])
p.show_grid()
p.show_axes()

<a alt="comandos de iluminação">light =</a> pv.Light(position = (-10, 1, 1), light_type = 'scene light')
p.add_light(light)
light = pv.Light(position = (10, 1, 1), light_type = 'scene light')
p.add_light(light)

p.set_background('royalblue', top = 'aliceblue')
<a alt="inserção do ator na cena: objeto 3D">p.add_mesh</a>(mesh, color = 'Red', show_edges = True, edge_color = 'grey', ambient = 0.3, 
    diffuse = 0.5, specular = 0.5, specular_power = 15)

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-98a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-99.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod68', 'cd68')" onmouseout="outFunc('cd68')"><span class="tooltiptext" id="cd68">Copiar o código</span></button></div>Criação de uma cena com Pyvista:
<pre><code id="cod68">import pyvista
import pyvista as pv

filename = 'C:/dados/chopper.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()

p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_grid()
p.show_axes()

light = pv.Light(position = (-10, 1, 1), light_type = 'scene light')
p.add_light(light)
light = pv.Light(position = (10, 1, 1), light_type = 'scene light')
p.add_light(light)

p.set_background('royalblue', top = 'aliceblue')
<a alt="mapa de cor Greens">p.add_mesh</a>(mesh, cmap = 'Greens', scalars = mesh.points[:, 2], show_scalar_bar = False, 
    show_edges = True, edge_color = 'grey', ambient = 0.3, diffuse = 0.5, specular = 0.5, 
    specular_power = 15)

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-99a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-100.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-101.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod69', 'cd69')" onmouseout="outFunc('cd69')"><span class="tooltiptext" id="cd69">Copiar o código</span></button></div>Criação de sombras em objetos de uma cena com Pyvista:
<pre><code id="cod69">import pyvista
import pyvista as pv
import numpy as np

filename = 'C:/dados/chopper.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()

p = pv.Plotter(lighting = None, window_size = [1000, 1000])
p.show_axes()

<a alt="limites das coordenadas">maxx =</a> np.max(mesh.points[:, 0])
maxy = np.max(mesh.points[:, 1])
minx = np.min(mesh.points[:, 0])
miny = np.min(mesh.points[:, 1])
minz = np.min(mesh.points[:, 2])
maxz = np.max(mesh.points[:, 2])

<a alt="posição da luz no centro relativo a x e y, com altura maxz + 150">light =</a> pv.Light(position = [(maxx + minx)/2, (maxy + miny)/2, maxz + 150], 
    focal_point = [(maxx + minx)/2, (maxy + miny)/2, 0], show_actor = True, 
    positional = True, cone_angle = 45, exponent = 50, intensity = 30)
p.add_light(light)

p.set_background('royalblue', top = 'aliceblue')
p.add_mesh(mesh, color = 'Green', show_edges = False, ambient = 0.3, diffuse = 0.5, 
    specular = 1, specular_power = 15, opacity = 1, <a alt="metalicidade">metallic =</a> 0.3, <a alt="rugosidade">roughness =</a> 0.6, <a alt="reflexo">pbr =</a> True)

<a alt="plano usado para projeção da sombra">grid =</a> pv.Plane(i_size = 5*(maxx - minx), j_size = 2*(maxy + miny), 
    center = [(maxx + minx)/2, (maxy + miny)/2, minz - 10])
p.add_mesh(grid, color = 'white')
<a alt="sombras">p.enable_shadows()</a>

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-101a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-102.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-103.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod70', 'cd70')" onmouseout="outFunc('cd70')"><span class="tooltiptext" id="cd70">Copiar o código</span></button></div>Criação de poliedros em uma cena do Pyvista:
<pre><code id="cod70">import pyvista as pv

<a alt="poliedros de Platão">kinds =</a> ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']
<a alt="posições dos centros dos sólidos">centers =</a> [(-1, 0, 0), (-1, 1, 0), (-1, 2, 0), (0, 1.5, 0), (0, 0.5, 0)]

<a alt="comando para inserir os poliedros">solids =</a> [pv.PlatonicSolid(kind, radius = 0.4, center = center) for kind, 
    center in zip(kinds, centers)]
<a alt="cores dos sólidos">colors =</a> ['aqua', 'red', 'orange', 'yellow', 'white']

p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.set_background('royalblue', top = 'aliceblue')

<a alt="iluminação da cena">for ind, solid in enumerate(solids):</a>
    p.add_mesh(solid, colors[ind], ambient = 0.3, smooth_shading = True, show_edges = True,
    diffuse = 0.8, specular = 0.5, specular_power = 2)

<a alt="plano usado para projeção da sombra">p.add_floor</a>('-z', lighting = True, color = 'white', pad = 0.4)
p.show_axes()

<a alt="posição da luz">p.add_light</a>(pv.Light(position = (1, -1, 5), focal_point = (0, 0, 0), color = 'white', 
    intensity = 0.8))
p.enable_shadows()

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-103a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod71', 'cd71')" onmouseout="outFunc('cd71')"><span class="tooltiptext" id="cd71">Copiar o código</span></button></div>Inserção de uma superfície em uma cena do Pyvista:
<pre><code id="cod71">import pyvista
import pyvista as pv

filename = 'C:/dados/everest.obj'
reader = <a alt="leitura do arquivo OBJ">pyvista.get_reader(filename)</a>
mesh = reader.read()

p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_grid()
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 32)
p.add_light(light)

p.set_background('royalblue', top = 'white')
p.add_mesh(mesh, cmap = 'coolwarm_r', <a alt="coordenadas usadas para o mapa de cores">scalars =</a> mesh.points[:, 2], show_scalar_bar = False,
    ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15, pbr = True, 
    metallic = 0.5, roughness = 0.2)

p.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo6/everest.obj" target="_blank">Arquivo OBJ - Monte Everest</a></p>
  <p>&#x1f4ca; <a href="modulo6/palcoyo.obj" target="_blank">Arquivo OBJ - montanhas de Palcoyo</a></p>
  </details></div>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-104.png" loading="lazy"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo7">7. Câmera</summary>
  <p>Material da página 105 até a página 114.</p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-104.png" loading="lazy"/>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/0HnuLDxwu0Y" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-105.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod72', 'cd72')" onmouseout="outFunc('cd72')"><span class="tooltiptext" id="cd72">Copiar o código</span></button></div>Projeção ortogonal (paralela) do Pyvista:
<pre><code id="cod72">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
<a alt="projeção paralela ativa na cena">p.enable_parallel_projection()</a>
p.show_grid(color="grey")
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
    ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

p.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-106.png" loading="lazy"/>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-106a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod73', 'cd73')" onmouseout="outFunc('cd73')"><span class="tooltiptext" id="cd73">Copiar o código</span></button></div>Zoom e Clipping plane da cena do Pyvista:
<pre><code id="cod73">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
      	ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

<a alt="ajuste do zoom da câmera">p.camera.zoom(0.8)</a>
<a alt="ajuste do clipping plane">p.camera.clipping_range =</a> (1000, 2500)
print(p.camera.clipping_range)

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-106b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-107.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod74', 'cd74')" onmouseout="outFunc('cd74')"><span class="tooltiptext" id="cd74">Copiar o código</span></button></div>Ponto focal da câmera em uma cena do Pyvista:
<pre><code id="cod74">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
      	ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

<a alt="ajuste do ponto focal da câmera">p.camera.focal_point =</a> (300, 0, -250)

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-107a.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod75', 'cd75')" onmouseout="outFunc('cd75')"><span class="tooltiptext" id="cd75">Copiar o código</span></button></div>Ângulo de visualização da câmera do Pyvista:
<pre><code id="cod75">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
      	ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

<a alt="ajuste do ângulo de visualização da câmera">p.camera.view_angle =</a> 155.0

p.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-107b.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod76', 'cd76')" onmouseout="outFunc('cd76')"><span class="tooltiptext" id="cd76">Copiar o código</span></button></div>Posição da câmera em uma cena do Pyvista:
<pre><code id="cod76">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
      	ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

<a alt="ajuste da posição da câmera">p.camera.position =</a> (1800, 1800, 0)

p.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-108.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod77', 'cd77')" onmouseout="outFunc('cd77')"><span class="tooltiptext" id="cd77">Copiar o código</span></button></div>Ângulos elevation, azimuth e roll da câmera do Pyvista:
<pre><code id="cod77">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
p = pv.Plotter(lighting = 'none', window_size = [1000, 1000])
p.show_axes()

light = pv.Light(position = (10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light', intensity = 1.5)
p.add_light(light)

p.set_background('white', top = 'white')

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 2], show_scalar_bar = False,
      	ambient = 0.3, diffuse = 0.5, specular = 0.5, specular_power = 15)

<a alt="câmera perpendicular ao plano xy">p.camera_position =</a> 'xy'
<a alt="ajuste do ângulo elevation da câmera">p.camera.elevation =</a> 45
<a alt="ajuste do ângulo azimuth da câmera">p.camera.azimuth =</a> 0
<a alt="ajuste do ângulo roll da câmera">p.camera.roll =</a> 0

p.show()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-109.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/x4mJ9RzEA3Q " frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-110.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-111.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-112.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod78', 'cd78')" onmouseout="outFunc('cd78')"><span class="tooltiptext" id="cd78">Copiar o código</span></button></div>CubeMap em uma cena do Pyvista:
<pre><code id="cod78">import pyvista
import pyvista as pv

filename = 'C:/dados/galleon.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
<a alt="ajuste angular para encaixe do cubemap">mesh.rotate_x(-90.0)</a>

p = pv.Plotter()
p.show_axes()
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light')
p.add_light(light)

cubemap = pyvista.cubemap('C:/dados/cubemap')
<a alt="função de conversão para o fundo da cena">p.add_actor(cubemap.to_skybox())</a>
<a alt="inserção do fundo da cena">p.set_environment_texture(cubemap)</a>

p.add_mesh(mesh, cmap = 'GnBu_r', scalars = mesh.points[:, 1], show_scalar_bar = False, 
    diffuse = 0.9, <a alt="propriedades de materiais com reflexos">pbr = True, metallic = 0.8, roughness = 0.1</a>)

p.add_axes()
<a alt="ajuste do ângulo roll da câmera">p.camera.roll =</a> 0

p.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo7/cubemap.zip" target="_blank">Arquivos que formam o Cubemap do fundo da cena</a></p>
  </details></div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-112a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-113.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod79', 'cd79')" onmouseout="outFunc('cd79')"><span class="tooltiptext" id="cd79">Copiar o código</span></button></div>CubeMap em uma cena do Pyvista:
<pre><code id="cod79">import pyvista
import pyvista as pv

<a alt="dados do helicóptero">filename =</a> 'C:/dados/chopper.ply'
reader = pyvista.get_reader(filename)
mesh = reader.read()
mesh.rotate_x(-90.0)

p = pv.Plotter()
p.show_axes()
light = pv.Light(position = (-10, 1, 1), light_type = 'scene light')
p.add_light(light)

cubemap = pyvista.cubemap('C:/dados/cubemap1')
p.add_actor(cubemap.to_skybox())
p.set_environment_texture(cubemap)

p.add_mesh(mesh, cmap = 'Reds_r', scalars = mesh.points[:, 1], show_scalar_bar = False, 
    diffuse = 0.9, pbr = True, metallic = 0.8, roughness = 0.1)

p.add_axes()
p.camera.roll=0
p.show()
</code></pre></figcaption>
  <p>&#x1f4ca; <a href="modulo7/cubemap1.zip" target="_blank">Arquivos que formam o Cubemap do fundo da cena</a></p>
  </details></div>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-113a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo8">8. Realidade Virtual</summary>
  <p>Material da página 115 até a página 145.</p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-114.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/PKF6a0Q_n5s" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-115.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://aframe.io/" target="_blank">https://aframe.io/</a></p></details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-115a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-116.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-117.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="001" name="sl">
			   <label for="001"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod80', 'cd80')" onmouseout="outFunc('cd80')"><span class="tooltiptext" id="cd80">Copiar o código</span></button></div>Cena de RV com um cubo, sem imagem de fundo:
<pre><code id="cod80">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    <a alt="biblioteca a-frame de RV">&lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;</a>
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       <a alt="objeto: cubo">&lt;a-box</a> color="green" <a alt="posição na cena">position=</a>"0 2 -4" <a alt="rotação em torno dos eixos y e z">rotation=</a>"0 45 45" <a alt="dimensões do objeto">scale=</a>"2 2 3"&gt;&lt;/a-box&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="002" name="sl">
			   <label for="002"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo0.htm" title="Cena de RV com um cubo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV com um cubo, sem imagem de fundo.<br><a href="modulo8/exemplo0.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-117a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="003" name="sl">
			   <label for="003"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod81', 'cd81')" onmouseout="outFunc('cd81')"><span class="tooltiptext" id="cd81">Copiar o código</span></button></div>Cena de RV com um cubo, com fundo azul:
<pre><code id="cod81">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-box color="green" position="0 2 -4" rotation="0 45 45" scale="2 2 3"&gt;&lt;/a-box&gt;
       <a alt="definição da cor do fundo da cena">&lt;a-sky</a> color="#99ccff"&gt;&lt;/a-sky&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="004" name="sl">
			   <label for="004"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo1b.htm" title="Cena de RV com um cubo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV com um cubo, sem imagem de fundo.<br><a href="modulo8/exemplo1b.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-117b.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="005" name="sl">
			   <label for="005"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod82', 'cd82')" onmouseout="outFunc('cd82')"><span class="tooltiptext" id="cd82">Copiar o código</span></button></div>Cena de RV com um cubo, com ambientes forest e japan:
<pre><code id="cod82">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    <a alt="biblioteca com ambientes a-frame">&lt;script src="https://unpkg.com/aframe-environment-component/dist/aframe-environment-component.min.js"&gt;&lt;/script&gt;</a>
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-box color="green" position="0 2 -4" rotation="0 45 45" scale="2 2 3"&gt;&lt;/a-box&gt;
       <a alt="ambiente forest com 500 árvores">&lt;a-entity</a> environment="preset: forest; dressingAmount: 500"&gt;&lt;/a-entity&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="006a" name="sl">
			   <label for="006a"></label>
			   <img src="modulo8/imagens/wasd.png" loading="lazy"/>
			   <figcaption>Combinação das teclas WASD usada para a movimentação de personagens em cenas de Realidade Virtual e jogos 3D.</figcaption>
		   </li>
		   <li>
			   <input type="radio" id="006" name="sl">
			   <label for="006"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo1a.htm" title="Cena de RV com um cubo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV com um cubo, com ambiente forest.<br><a href="modulo8/exemplo1a.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="007" name="sl">
			   <label for="007"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo1.htm" title="Cena de RV com um cubo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV com um cubo, com ambiente japan. Modifique a tag do ambiente no código para inserir o cubo em outros ambientes do a-frame.<br><a href="modulo8/exemplo1.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/gZXwqvR7WY4" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-118.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="008" name="sl">
			   <label for="008"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod83', 'cd83')" onmouseout="outFunc('cd83')"><span class="tooltiptext" id="cd83">Copiar o código</span></button></div>Cena de RV com imagem equiretangular de fundo:
<pre><code id="cod83">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       <a alt="tag para inserir as referências de imagens e objetos">&lt;a-assets&gt;</a>
          <a alt="caminhos das imagens das texturas e fundo da cena">&lt;img id="ceu"</a> src="./imagens/equi1.jpg"&gt;
          &lt;img id="textura1" src="./imagens/textura1.jpg"&gt;
          &lt;img id="textura2" src="./imagens/textura2.jpg"&gt;  
       &lt;/a-assets&gt;
       <a alt="imagem textura1 no cubo">&lt;a-box src="#textura1"</a> position="0 1 -4" rotation="0 45 45" scale="1 1 1.5"&gt;&lt;/a-box&gt;
       <a alt="imagem textura2 no cilindro">&lt;a-cylinder src="#textura2"</a> position="2.5 1 -4" radius="0.5" height="2"&gt;&lt;/a-cylinder&gt;
       <a alt="imagem equiretangular no céu da cena">&lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;</a>
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="009" name="sl">
			   <label for="009"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo2.htm" title="Cena de RV com um cubo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV com um cubo, com imagem equiretangular de fundo.<br><a href="modulo8/exemplo2.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-119.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="010" name="sl">
			   <label for="010"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod84', 'cd84')" onmouseout="outFunc('cd84')"><span class="tooltiptext" id="cd84">Copiar o código</span></button></div>Cena de RV da representação da Terra e da Lua:
<pre><code id="cod84">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-assets&gt;
          &lt;img id="ceu" src="./imagens/2k_stars_milky_way.jpg"&gt;
          &lt;img id="textura1" src="./imagens/2k_earth_daymap.jpg"&gt;
          &lt;img id="textura2" src="./imagens/2k_moon.jpg"&gt;  
       &lt;/a-assets&gt;
       <a alt="esfera que representa a Terra">&lt;a-sphere src="#textura1"</a> position="0 2 -4" scale="2 2 2"&gt;&lt;/a-sphere&gt;
       <a alt="esfera que representa a Lua">&lt;a-sphere src="#textura2"</a> position="4 3 -4" scale="0.5 0.5 0.5"&gt;&lt;/a-sphere&gt;
       <a alt="céu da cena com imagem equiretangular da via Láctea">&lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;</a>
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="011" name="sl">
			   <label for="011"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo3.htm" title="Cena de RV com a Terra e a Lua" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena de RV da representação da Terra e da Lua.<br><a href="modulo8/exemplo3.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-119a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/-hE0FfxyLqg" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-120.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="012" name="sl">
			   <label for="012"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod85', 'cd85')" onmouseout="outFunc('cd85')"><span class="tooltiptext" id="cd85">Copiar o código</span></button></div>Iluminação ambiente em uma cena de RV:
<pre><code id="cod85">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" &gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4"&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" &gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz ambiente branca com intensidade 0.8">&lt;a-light type="ambient"</a> color="white" intensity="0.8"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="013" name="sl">
			   <label for="013"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo4a.htm" title="Iluminação ambiente" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação ambiente em uma cena de RV.<br><a href="modulo8/exemplo4a.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-120a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="014" name="sl">
			   <label for="014"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod86', 'cd86')" onmouseout="outFunc('cd86')"><span class="tooltiptext" id="cd86">Copiar o código</span></button></div>Iluminação direcional em uma cena de RV:
<pre><code id="cod86">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" &gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4"&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" &gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz direcional com intensidade 1.5">&lt;a-light type="directional"</a> intensity="1.5" <a alt="posição da fonte de luz">position=</a>"3 3 3" <a alt="referência do alvo">target=</a>"#directionaltarget"&gt;
          <a alt="posição do alvo da fonte de luz">&lt;a-entity id="directionaltarget"</a> position="-1 -1 -1"&gt;&lt;/a-entity&gt;
       &lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="015" name="sl">
			   <label for="015"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo4b.htm" title="Iluminação ambiente" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação direcional em uma cena de RV.<br><a href="modulo8/exemplo4b.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-120b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-121.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="016" name="sl">
			   <label for="016"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod87', 'cd87')" onmouseout="outFunc('cd87')"><span class="tooltiptext" id="cd87">Copiar o código</span></button></div>Iluminação direcional com 3 fontes:
<pre><code id="cod87">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
          <a alt="objeto que recebe projeção de sombra">shadow="receive: true"</a>&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4"
          <a alt="objeto que recebe projeção de sombra">shadow="receive: true"</a>&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4"
          <a alt="objeto que recebe projeção de sombra">shadow="receive: true"</a>&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" 
          <a alt="objeto que produz projeção de sombra">shadow="cast: true"</a>&gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz direcional com intensidade 0.8">&lt;a-light type="directional" intensity="0.8"</a> position="0 5 -4" light="castShadow:true" 
         target="#directionaltargetY"&gt;
            <a alt="alvo na direção do eixo y">&lt;a-entity id="directionaltargetY"</a> position="0 -1 0"&gt;&lt;/a-entity&gt;
       &lt;/a-light&gt;
       <a alt="luz direcional com intensidade 0.8">&lt;a-light type="directional" intensity="0.8"</a> position="0 0 2" light="castShadow:true" 
         target="#directionaltargetZ"&gt;
            <a alt="alvo na direção do eixo z">&lt;a-entity id="directionaltargetZ"</a> position="0 0 -1"&gt;&lt;/a-entity&gt;
       &lt;/a-light&gt;
       <a alt="luz direcional com intensidade 0.8">&lt;a-light type="directional" intensity="0.8"</a> position="5 0 -4" light="castShadow:true" 
         target="#directionaltargetX"&gt;
            <a alt="alvo na direção do eixo x">&lt;a-entity id="directionaltargetX"</a> position="-1 0 0"&gt;&lt;/a-entity&gt;
       &lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="017" name="sl">
			   <label for="017"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo4.htm" title="Iluminação direcional" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação direcional com 3 fontes.<br><a href="modulo8/exemplo4.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-121a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-122.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="018" name="sl">
			   <label for="018"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod88', 'cd88')" onmouseout="outFunc('cd88')"><span class="tooltiptext" id="cd88">Copiar o código</span></button></div>Iluminação hemisférica:
<pre><code id="cod88">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene <a alt="suavização das projeções de sombras">shadow="type: pcfsoft"&gt;</a>
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" 
          shadow="cast: true"&gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz hemisférica com intensidade 0.7">&lt;a-light type="hemisphere"</a> color=<a alt="cor da luz">"#eaeaea"</a> light="<a alt="cor da luz projetada no piso">groundColor:</a> green" 
          intensity="0.7"&gt;&lt;/a-light&gt;
       <a alt="luz direcional com intensidade 0.5">&lt;a-light type="directional"</a> intensity="0.5" position="8 5 5" light="castShadow:true"
         target="#directionaltargetZ"&gt;
             &lt;a-entity id="directionaltargetZ" position="-1.3 -1 -1"&gt;&lt;/a-entity&gt;
       &lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="019" name="sl">
			   <label for="019"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo5.htm" title="Iluminação hemisférica" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação hemisférica.<br><a href="modulo8/exemplo5.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-122a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="020" name="sl">
			   <label for="020"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod89', 'cd89')" onmouseout="outFunc('cd89')"><span class="tooltiptext" id="cd89">Copiar o código</span></button></div>Iluminação ponto:
<pre><code id="cod89">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" 
          shadow="cast: true"&gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz hemisférica com intensidade 0.7">&lt;a-light type="hemisphere"</a> color="#eaeaea" light="groundColor:green" 
          intensity="0.7"&gt;&lt;/a-light&gt;
       <a alt="luz point com intensidade 0.75">&lt;a-light type="point"</a> intensity="0.75" <a alt="distância de 50 metros">distance=</a>"50" <a alt="fator de decaimento 7">decay=</a>"7" position="0 3 0" 
          light="castShadow: true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="021" name="sl">
			   <label for="021"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo6.htm" title="Iluminação ponto" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação ponto.<br><a href="modulo8/exemplo6.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-122b.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="022" name="sl">
			   <label for="022"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod90', 'cd90')" onmouseout="outFunc('cd90')"><span class="tooltiptext" id="cd90">Copiar o código</span></button></div>Iluminação spot:
<pre><code id="cod90">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4" 
          shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-box color="#F7819F" position="0 2 -4" rotation="0 45 45" scale="2 2 2" 
          shadow="cast: true"&gt;&lt;/a-box&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       <a alt="luz hemisférica com intensidade 0.7">&lt;a-light type="hemisphere"</a> color="#eaeaea" light="groundColor:green" 
          intensity="0.7"&gt;&lt;/a-light&gt;
       <a alt="luz spot com intensidade 0.75">&lt;a-light type="spot" intensity=</a>"0.75" <a alt="ângulo de abertura">angle=</a>"45" <a alt="fator de suavização da sombra">penumbra=</a>"0.2" light="castShadow:true" 
          position="0 2 -0.5"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="023" name="sl">
			   <label for="023"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo7.htm" title="Iluminação spot" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Iluminação spot.<br><a href="modulo8/exemplo7.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-123.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/NRAsCe9N0c4" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-124.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="024" name="sl">
			   <label for="024"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod91', 'cd91')" onmouseout="outFunc('cd91')"><span class="tooltiptext" id="cd91">Copiar o código</span></button></div>Animação de uma esfera em torno do eixo z:
<pre><code id="cod91">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-entity position="-1 0.5 -4"&gt;
          <a alt="cilindros que representam os eixos">&lt;a-cylinder</a> radius="0.02" height="3" position="0 0.5 0" 
             color="rgb(0,255,0)"&gt;&lt;/a-cylinder&gt;
          &lt;a-cylinder rotation="0 0 90" radius="0.02" height="3" position="0.5 0 0" 
             color="rgb(255,0,0)"&gt;&lt;/a-cylinder&gt;
          &lt;a-cylinder rotation="90 0 0" radius="0.02" height="3" position="0 0 0.5" 
             color="rgb(0,0,255)"&gt;&lt;/a-cylinder&gt;
          <a alt="textos dos rótulos dos eixos">&lt;a-text</a> position="0.05 -0.1 0" value="O" width="4" color="black"&gt;&lt;/a-text&gt;
          &lt;a-text position="2 0 0" value="x" width="4" color="black"&gt;&lt;/a-text&gt;
          &lt;a-text position="0 2 0" value="y" width="4" color="black"&gt;&lt;/a-text&gt;
          &lt;a-text position="0 0 2" value="z" width="4" color="black"&gt;&lt;/a-text&gt;
          <a alt="toro para mostrar a trajetória da animação">&lt;a-torus</a> position="0 0 1"radius="1.1" radius-tubular="0.01" segments-tubular="100" 
             opacity="0.2"&gt;&lt;/a-torus&gt;
          <a alt="animação da esfera">&lt;a-entity animation="property: rotation;</a> to: 0 0 360; loop: true; 
            dir: alternate; dur: 10000;"&gt;
               &lt;a-sphere <a alt="posição inicial da esfera">position=</a>"0.5 1 1" radius="0.1" color="rgb(200,30,100)" &gt;&lt;/a-sphere&gt;
          &lt;/a-entity&gt;
       &lt;/a-entity&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="025" name="sl">
			   <label for="025"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo8a.htm" title="Animação em torno de um eixo" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Animação da esfera em torno do eixo z.<br><a href="modulo8/exemplo8a.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-124a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-125.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="026" name="sl">
			   <label for="026"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod92', 'cd92')" onmouseout="outFunc('cd92')"><span class="tooltiptext" id="cd92">Copiar o código</span></button></div>Animação da Lua em torno do centro da Terra:
<pre><code id="cod92">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-assets&gt;
          &lt;img id="ceu" src="./imagens/2k_stars_milky_way.jpg"&gt;
          &lt;img id="textura1" src="./imagens/2k_earth_daymap.jpg"&gt;
          &lt;img id="textura2" src="./imagens/2k_moon.jpg"&gt;  
       &lt;/a-assets&gt;
       &lt;a-entity position="0 2 -4" rotation="0 0 30"&gt; 
          &lt;a-entity rotation="0 0 -30"&gt;
              &lt;a-sphere src="#textura1" scale="2 2 2" <a alt="animação da Terra em torno do próprio eixo">animation="property:</a> rotation; from: 0 0 0; 
                to: 0 360 0; loop: true; dur: 10000; easing: linear;"&gt;&lt;/a-sphere&gt;
          &lt;/a-entity&gt;
          &lt;a-entity <a alt="animação da Lua em torno do centro da Terra">animation="property:</a> rotation; from: 0 360 0; to: 0 0 0; loop: true; 
            dur: 5000; easing: linear"&gt;
              &lt;a-sphere <a alt="posição relativa">position="3 0 0"</a> src="#textura2" scale="0.5 0.5 0.5" &gt;&lt;/a-sphere&gt;
          &lt;/a-entity&gt;
       &lt;/a-entity&gt;
       &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
       &lt;a-light type="hemisphere" color="#eaeaea" light="groundColor:green" 
         intensity="0.7"&gt;&lt;/a-light&gt;
       &lt;a-light type="point" intensity="0.75" distance="50" decay="2" position="0 3 0" 
         light="castShadow: true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="027" name="sl">
			   <label for="027"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo8.htm" title="Animação de duas esferas" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Animação da Lua em torno do centro da Terra.<br><a href="modulo8/exemplo8.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-125a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="028" name="sl">
			   <label for="028"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod93', 'cd93')" onmouseout="outFunc('cd93')"><span class="tooltiptext" id="cd93">Copiar o código</span></button></div>Animação com mudanças de cores:
<pre><code id="cod93">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-icosahedron color="blue" position="1 1.5 -4" radius="1.5" 
         <a alt="mudança de cor do icosaedro em 5 segundos">animation=</a>"property: components.material.material.color; type: color; to: red; 
         loop: true; dir: alternate; dur: 5000;"&gt;&lt;/a-icosahedron&gt;
       &lt;a-sky color="aliceblue" <a alt="mudança de cor do fundo da cena em 7 segundos">animation=</a>"property: components.material.material.color; 
         type: color; to: aqua; loop: true; dir: alternate; dur: 7000;"&gt;&lt;/a-sky&gt;
       &lt;a-light type="hemisphere" color="#eaeaea" light="groundColor:green" 
         intensity="0.7"&gt;&lt;/a-light&gt;
       &lt;a-light type="point" intensity="0.75" distance="50" decay="2" position="0 3 0" 
         light="castShadow: true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="029" name="sl">
			   <label for="029"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo9.htm" title="Animação de mudança de cor" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Animação com mudanças de cores.<br><a href="modulo8/exemplo9.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-125b.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="030" name="sl">
			   <label for="030"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod94', 'cd94')" onmouseout="outFunc('cd94')"><span class="tooltiptext" id="cd94">Copiar o código</span></button></div>Animação com mudanças de cores:
<pre><code id="cod94">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-cone color="royalblue" position="0 2 -3" rotation="0 0 45" radius-bottom="0.75" 
         height="2.5" shadow="cast: true"&gt;&lt;/a-cone&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       &lt;a-light type="hemisphere" color="#eaeaea" light="groundColor:green" intensity="0.7" 
         <a alt="mudança de intensidade da luz hemisférica em 5 segundos">animation="property:</a> intensity; to: 0.2; loop: true; dir: alternate; dur: 5000;"&gt;&lt;/a-light&gt;
       &lt;a-light type="spot" intensity="0.75" angle="45" penumbra="0.2" light="castShadow:true" 
         position="0 2 -0.5"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="031" name="sl">
			   <label for="031"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo10.htm" title="Animação de mudança de intensidade da luz" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Animação de intensidade de luz.<br><a href="modulo8/exemplo10.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-126.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="032" name="sl">
			   <label for="032"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod95', 'cd95')" onmouseout="outFunc('cd95')"><span class="tooltiptext" id="cd95">Copiar o código</span></button></div>Animação com mudanças de cores:
<pre><code id="cod95">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-plane color="#A9F5D0" position="0 2 -6" width="8" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="0 0 -4" rotation="-90 0 0" width="8" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-plane color="#A9F5D0" position="-4 2 -4" rotation="0 90 0" width="4" height="4" 
         shadow="receive: true"&gt;&lt;/a-plane&gt;
       &lt;a-cone color="royalblue" position="0 2 -3" rotation="0 0 45" radius-bottom="0.75" 
         height="2.5" shadow="cast: true"&gt;&lt;/a-cone&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       &lt;a-light type="hemisphere" color="#eaeaea" light="groundColor:green" 
         intensity="0.7"&gt;&lt;/a-light&gt;
       &lt;a-light type="spot" intensity="0.75" angle="45" penumbra="0.2" light="castShadow:true" 
         position="-2 2 -0.5" <a alt="mudança da posição da luz spot em 10 segundos">animation="property:</a> position; to: 2 2 -0.5; loop: true;
         dir: alternate; dur: 10000;"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="033" name="sl">
			   <label for="033"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo11.htm" title="Animação de mudança de posição da luz" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Animação da posição da fonte de luz.<br><a href="modulo8/exemplo11.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-126a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-127.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="034" name="sl">
			   <label for="034"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod96', 'cd96')" onmouseout="outFunc('cd96')"><span class="tooltiptext" id="cd96">Copiar o código</span></button></div>Propriedades da câmera:
<pre><code id="cod96">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
       &lt;a-assets&gt;
           &lt;img id="arvore" src="./imagens/treebark.png"&gt;
       &lt;/a-assets&gt;
       <a alt="posição da câmera">&lt;a-camera</a> position="0 2 2"&gt;&lt;/a-camera&gt;
       &lt;a-cylinder src="#arvore" position="0 2 0" radius="0.5" height="2" 
         <a alt="propriedades do material">metalness="0.6" roughness="0.3"</a> side="double"&gt;&lt;/a-cylinder&gt;
       &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
       &lt;a-light type="ambient" color="white" intensity="0.4"&gt;&lt;/a-light&gt;
       &lt;a-light type="directional" intensity="0.8" position="-1 0 0"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="035" name="sl">
			   <label for="035"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo12.htm" title="Posição da câmera" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Posição da câmera.<br><a href="modulo8/exemplo12.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-127a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/hcGsdK-VYuM" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-128.png" loading="lazy"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://jaxry.github.io/panorama-to-cubemap/" target="_blank">https://jaxry.github.io/panorama-to-cubemap/</a></p></details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-128a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="036" name="sl">
			   <label for="036"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod97', 'cd97')" onmouseout="outFunc('cd97')"><span class="tooltiptext" id="cd97">Copiar o código</span></button></div>Órbita da câmera e reflexão da imagem de fundo nos objetos:
<pre><code id="cod97">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    <a alt="biblioteca de órbita da câmera">&lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;</a>
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection&gt;
        &lt;a-assets&gt;
           &lt;img id="ceu" src="./imagens/equi1.jpg"&gt;
           &lt;img id="metal" src="./imagens/metal1.png"&gt;
           <a alt="imagens que formam o fundo da cena: formato cubemap">&lt;a-cubemap id="ceu2"&gt;</a>
              &lt;img src="./imagens/cubemap/Right.png"&gt;
              &lt;img src="./imagens/cubemap/Left.png"&gt;
              &lt;img src="./imagens/cubemap/Top.png"&gt;
              &lt;img src="./imagens/cubemap/Bottom.png"&gt;
              &lt;img src="./imagens/cubemap/Front.png"&gt;
              &lt;img src="./imagens/cubemap/Back.png"&gt;
           &lt;/a-cubemap&gt;
        &lt;/a-assets&gt;
        &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
        <a alt="câmera com propriedades de órbita">&lt;a-camera orbit-controls</a>="<a alt="alvo">target:</a> -1 1.5 1; minDistance: 0.5; maxDistance: 180; 
          <a alt="posição inicial da câmera">initialPosition:</a> -1 1.6 3.5"&gt;&lt;/a-camera&gt;
        &lt;a-sphere position="1 2 0.5" radius="1" side="double" color="silver" 
          metalness="1" roughness="0" segments-height="36" shadow="" segments-width="64" 
          <a alt="cubemap para reflexão">material="envMap:</a> #ceu2;"&gt;&lt;/a-sphere&gt;
        &lt;a-sphere position="-2 1.5 -0.5" color="green" radius="1" side="double" 
          metalness="1" roughness="0" segments-height="36" shadow="" segments-width="64" 
          <a alt="cubemap para reflexão">material=</a>"envMap: #ceu2;"&gt;&lt;/a-sphere&gt;		
        &lt;a-cylinder src="#metal" position="-3 0.5 1.5" color="white" radius="0.5"
          height="1.5" side="double" metalness="1" roughness="0" shadow="" 
          <a alt="cubemap para reflexão">material=</a>"envMap: #ceu2;"&gt;&lt;/a-cylinder&gt;		
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.75" angle="60" penumbra="0.5" 
          shadow="cast: true; receive: false" position="-2 2 4"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="037" name="sl">
			   <label for="037"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo13.htm" title="Posição da câmera" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Órbita da câmera e reflexão do fundo da cena nos objetos.<br><a href="modulo8/exemplo13.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-129.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="038" name="sl">
			   <label for="038"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod98', 'cd98')" onmouseout="outFunc('cd98')"><span class="tooltiptext" id="cd98">Copiar o código</span></button></div>Órbita da câmera e reflexão da imagem de fundo e entre objetos:
<pre><code id="cod98">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    <a alt="versão aframe com suporte de reflexões entre objetos">&lt;script src="https://aframe.io/releases/1.0.4/aframe.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de reflexões entre objetos">&lt;script src="./java/camera-cube-env.js"&gt;&lt;/script&gt;</a>
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection&gt;
        &lt;a-assets&gt;
           &lt;img id="ceu" src="./imagens/equi1.jpg"&gt;
           &lt;img id="metal" src="./imagens/metal1.png"&gt;
        &lt;/a-assets&gt;
        &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
        &lt;a-camera orbit-controls="target: -1 1.5 1; minDistance: 0.5; maxDistance: 180; 
          initialPosition: -1 1.6 3.5"&gt;&lt;/a-camera&gt;
        &lt;a-sphere position="1 2 0.5" radius="1" side="double" color="silver" 
          metalness="1" roughness="0" segments-height="36" shadow="" segments-width="64" 
          <a alt="habilita a reflexão entre objetos da cena">camera-cube-env=</a>"distance: 500; resolution: 512; repeat: true; interval: 1;"&gt;&lt;/a-sphere&gt;
        &lt;a-sphere position="-2 1.5 -0.5" color="green" radius="1" side="double" 
          metalness="1" roughness="0" segments-height="36" shadow="" segments-width="64" 
          <a alt="habilita a reflexão entre objetos da cena">camera-cube-env=</a>"distance: 500; resolution: 512; repeat: true; interval: 1;"&gt;&lt;/a-sphere&gt;
        &lt;a-cylinder src="#metal" position="-3 0.5 1.5" color="white" radius="0.5"
          height="1.5" side="double" metalness="1" roughness="0" shadow="" 
          <a alt="habilita a reflexão entre objetos da cena">camera-cube-env=</a>"distance: 500; resolution: 512; repeat: true; interval: 1;"&gt;&lt;/a-cylinder&gt;
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.75" angle="60" penumbra="0.5" 
          shadow="cast: true; receive: false" position="-2 2 4"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="039" name="sl">
			   <label for="039"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo14.htm" title="Posição da câmera" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Órbita da câmera e reflexão do fundo e entre os objetos da cena.<br><a href="modulo8/exemplo14.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-129a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/r7XGkJEqSds" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-130.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="040" name="sl">
			   <label for="040"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod99', 'cd99')" onmouseout="outFunc('cd99')"><span class="tooltiptext" id="cd99">Copiar o código</span></button></div>Representação de poliedro com sombra em um plano:
<pre><code id="cod99">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: 0 2 0; minDistance: 0.5; maxDistance: 180; 
          initialPosition: -1 1.6 4.5"&gt;&lt;/a-entity&gt;
        &lt;a-octahedron position="0 2 0" radius="2" color="black" 
          <a alt="representação aramada: arestas">wireframe=</a>"true" <a alt="habilita a projeção de sombra">shadow=</a>"cast:true"&gt;&lt;/a-octahedron&gt;
        <a alt="representação das faces">&lt;a-octahedron</a> position="0 2 0" radius="2" side="double"
          color="tomato" metalness="0.6" roughness="0.3" opacity="0.5"&gt;&lt;/a-octahedron&gt;
        &lt;a-plane color="#A9F5D0" position="0 -0.5 0" rotation="-90 0 0" width="5" height="5"
          <a alt="habilita a projeção de sombra">shadow=</a>"receive: true;"&gt;&lt;/a-plane&gt;
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" intensity="0.3" position="0 5 0" 
          <a alt="luz que produz as sombras na cena">light=</a>"castShadow:true" target="#directionaltargetY"&gt;
            &lt;a-entity id="directionaltargetY" position="0 -1 0"&gt;&lt;/a-entity&gt;
        &lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="041" name="sl">
			   <label for="041"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo15.htm" title="Sombras de poliedros" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Octaedro com sombras projetadas em um plano, produzidas por meio de luz direcional.<br><a href="modulo8/exemplo15.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-130a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-131.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="042" name="sl">
			   <label for="042"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod100', 'cd100')" onmouseout="outFunc('cd100')"><span class="tooltiptext" id="cd100">Copiar o código</span></button></div>Propriedades comuns de objetos agrupadas com a tag a-mixin:
<pre><code id="cod100">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection&gt;
        &lt;a-assets&gt;
           &lt;img id="ceu" src="./imagens/equi1.jpg"&gt;
           &lt;img id="metal" src="./imagens/metal1.png"&gt;
           &lt;a-cubemap id="ceu2"&gt;
              &lt;img src="./imagens/cubemap/Right.png"&gt;
              &lt;img src="./imagens/cubemap/Left.png"&gt;
              &lt;img src="./imagens/cubemap/Top.png"&gt;
              &lt;img src="./imagens/cubemap/Bottom.png"&gt;
              &lt;img src="./imagens/cubemap/Front.png"&gt;
              &lt;img src="./imagens/cubemap/Back.png"&gt;
           &lt;/a-cubemap&gt;
           &lt;a-mixin <a alt="propriedades de reflexões e materiais">id="padrao"</a> material="metalness:1; roughness:0; side:double; envMap: #ceu2;" 
             shadow=""&gt;&lt;/a-mixin&gt;
           &lt;a-mixin <a alt="propriedades geométricas">id="padrao2"</a> geometry="segments-height:36; segments-width:64; radius:1;"&gt;&lt;/a-mixin&gt;
        &lt;/a-assets&gt;
        &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: -1 1.5 1; minDistance: 0.5; maxDistance: 180; 
          initialPosition: -1 1.6 3.5"&gt;&lt;/a-entity&gt;
        &lt;a-sphere position="1 2 0.5" color="silver" <a alt="atribuição das propriedades">mixin="padrao padrao2"</a>&gt;&lt;/a-sphere&gt;
        &lt;a-sphere position="-2 1.5 -0.5" color="green" <a alt="atribuição das propriedades">mixin="padrao padrao2"</a>&gt;&lt;/a-sphere&gt;
        &lt;a-cylinder src="#metal" position="-3 0.5 1.5" color="white" radius="0.5" 
          height="1.5" <a alt="atribuição das propriedades">mixin="padrao"</a>&gt;&lt;/a-cylinder&gt;
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.75" angle="60" penumbra="0.5" 
          shadow="cast: true; receive: false" position="-2 2 4"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="043" name="sl">
			   <label for="043"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo16.htm" title="Tag mixin de propriedades" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Propriedades agrupadas na tag a-mixin.<br><a href="modulo8/exemplo16.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-131a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-132.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="044" name="sl">
			   <label for="044"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod101', 'cd101')" onmouseout="outFunc('cd101')"><span class="tooltiptext" id="cd101">Copiar o código</span></button></div>Construção de um poliedro usando a propriedade faceset:
<pre><code id="cod101">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    <a alt="referência da biblioteca com suporte do faceset">&lt;script src="https://aframe.io/releases/1.1.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
    <a alt="referência da biblioteca faceset">&lt;script src="https://andreasplesch.github.io/aframe-faceset-component/dist/aframe-faceset-component.min.js"&gt;&lt;/script&gt;</a>
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene&gt;
        &lt;a-assets&gt;
            &lt;a-mixin <a alt="propriedade das arestas da pirâmide">id="aramado"</a> material="color: red; wireframe: true; wireframe-linewidth:1;"&gt;&lt;/a-mixin&gt;
            &lt;a-mixin <a alt="padrão da cor da pirâmide">id="cor1"</a> material="color: #d8ef09"&gt;&lt;/a-mixin&gt;
            &lt;a-mixin <a alt="propriedades de materiais">id="padrao"</a> material="opacity: 0.5; side: double; metalness:0.3; roughness:0.9;"&gt;&lt;/a-mixin&gt;
            &lt;a-mixin <a alt="propriedades dos rótulos">id="texto"</a> text="width: 4; side:double; color: black" rotation="-25 0 0"&gt;
        &lt;/a-assets&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: 0 1 0.7; minDistance: 0.5; maxDistance: 180; 
          initialPosition: 0 1.6 3"&gt;&lt;/a-entity&gt;
        &lt;a-entity <a alt="tag para agrupar os elementos do poliedro">id="piramide_pentagonal_reta"</a>&gt;
           <a alt="rótulos dos vértices">&lt;a-text</a> position="-0.2 0 0" value="A" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-text position="1.1 0 0" value="B" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-text position="1.31 0 0.95" value="C" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-text position="0.5 0 1.5" value="D" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-text position="-0.45 0 0.95" value="E" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-text position="0.5 2.1 0.7" value="V" mixin="texto"&gt;&lt;/a-text&gt;
           &lt;a-entity <a alt="sequência dos vértices da base da pirâmide">faceset=</a>"vertices: 0 0 0  1 0 0  1.31 0 0.95  0.5 0 1.5  -0.31 0 0.95" 
             mixin="cor1 padrao"&gt;&lt;/a-entity&gt;
           &lt;a-entity <a alt="sequência dos vértices das faces laterais">faceset=</a>"vertices: 0 0 0  1 0 0  1.31 0 0.95  0.5 0 1.5  -0.31 0 0.95  0.5 2 0.7;
             <a alt="projeção ortogonal ao eixo y">projectdir:y"</a> mixin="cor1 padrao"&gt;&lt;/a-entity&gt;
           &lt;a-entity <a alt="arestas do poliedro">faceset=</a>"vertices: 0 0 0  1 0 0  1.31 0 0.95  0.5 0 1.5  -0.31 0 0.95  0.5 2 0.7; 
             projectdir:y" mixin="aramado"&gt;&lt;/a-entity&gt;
           &lt;a-entity <a alt="linhas que representam a altura e a projeção da aresta VD">line=</a>"start: 0.5 2 0.7; end: 0.5 0 0.7; color: white" 
             line__2="start: 0.5 0 0.7; end: 0.5 0 1.5; color: white"&gt;&lt;/a-entity&gt;
        &lt;/a-entity&gt;
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.6" position="1 1 3" light="castShadow:true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="045" name="sl">
			   <label for="045"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo17.htm" title="Representação de uma pirâmide" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Representação de uma pirâmide com rótulos nos vértices, altura e projeção de uma aresta na base.<br><a href="modulo8/exemplo17.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-132a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/uHsW9hxpWmM" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-133.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-134.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-135.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="046" name="sl">
			   <label for="046"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod102', 'cd102')" onmouseout="outFunc('cd102')"><span class="tooltiptext" id="cd102">Copiar o código</span></button></div>Interação com objetos de uma cena (raio laser):
<pre><code id="cod102">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;title&gt;Interações com objetos de uma cena&lt;/title&gt;
    <a alt="referência da biblioteca com suporte de interações">&lt;script src="https://aframe.io/releases/1.1.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de controles para movimentar objetos">&lt;script src="https://unpkg.com/super-hands/dist/super-hands.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de propriedades físicas">&lt;script src="https://rawgit.com/donmccurdy/aframe-physics-system/v4.0.0/dist/aframe-physics-system.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de interação">&lt;script src="https://unpkg.com/aframe-event-set-component@^4.1.1/dist/aframe-event-set-component.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca com propriedades físicas adicionais">&lt;script src="https://unpkg.com/aframe-physics-extras/dist/aframe-physics-extras.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de teleporte">&lt;script src="https://fernandojsg.github.io/aframe-teleport-controls/dist/aframe-teleport-controls.min.js"&gt;&lt;/script&gt;</a>
    &lt;script&gt; <a alt="função de melhoria das propriedades físicas">AFRAME.registerComponent('phase-shift', {</a>
      init: function () {
        var el = this.el
        el.addEventListener('gripdown', function () {
           el.setAttribute('collision-filter', {collisionForces: true})
        })
        el.addEventListener('gripup', function () {
           el.setAttribute('collision-filter', {collisionForces: false})
        })
      }
    });
    &lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene physics shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
           &lt;a-mixin id="cubo" geometry="width: 0.5; height: 0.5; depth: 0.5;" shadow 
             <a alt="propriedades para manipulação dos cubos">hoverable grabbable stretchable draggable droppable</a> event-set__hoveron="_event: hover-start; 
             material.opacity: 0.7; transparent: true" event-set__hoveroff="_event: hover-end; 
             material.opacity: 1; transparent: false" <a alt="propriedades físicas dos cubos">dynamic-body=</a>"linearDamping:0.1; 
             angularDamping:0.8; mass:0.5;"&gt;&lt;/a-mixin&gt;
           &lt;a-mixin <a alt="propriedades do piso">id="piso"</a> geometry="width: 10; height: 0.3; depth: 10;" static-body shadow&gt;&lt;/a-mixin&gt;
	   &lt;a-mixin <a alt="propriedades para o ponteiro de interação com objetos">id="pointer" phase-shift</a> raycaster="showLine: true; objects: .cubo, a-box" 
	     collision-filter="collisionForces: false" static-body="shape: sphere; sphereRadius: 0.02" 
	     super-hands="colliderEvent: raycaster-intersection; colliderEventProperty: els;  
	     colliderEndEvent:raycaster-intersection-cleared; colliderEndEventProperty: clearedEls;"&gt;&lt;/a-mixin&gt;
	   &lt;a-mixin <a alt="controle para manipular objetos">id="controle" mixin="pointer"</a> hand-controls="hand: left"&gt;&lt;/a-mixin&gt;
        &lt;/a-assets&gt;
        &lt;a-entity <a alt="interações conectadas com a câmera">id="cameraRig"</a>&gt;
           &lt;a-camera id="head" look-controls wasd-controls position="0 1 2" 
             <a alt="interação com o cursor do mouse">cursor="rayOrigin:mouse"</a> static-body="shape: sphere; sphereRadius: 0.001" 
             <a alt="propriedades de colisão e rastreamento">super-hands=</a>"colliderEvent: raycaster-intersection; colliderEventProperty: els;
             colliderEndEvent:raycaster-intersection-cleared; colliderEndEventProperty: clearedEls;"&gt;&lt;/a-camera&gt;
           &lt;a-entity <a alt="teleporte nos cubos e no piso">teleport-controls=</a>"cameraRig: #cameraRig; collisionEntities: [mixin='cubo'],
             [mixin='piso']; teleportOrigin: #head; button: grip;" <a alt="controle com a mão direita">hand-controls="hand: right"</a>
             gearvr-controls daydream-controls&gt;&lt;/a-entity&gt;
           &lt;a-entity <a alt="controle de manipulação da mão esquerda">id="lhand" mixin="controle"</a> super-hands&gt;&lt;/a-entity&gt;
        &lt;/a-entity&gt;
        <a alt="posições dos objetos na cena">&lt;a-box mixin="piso"</a> class="piso" position="0 -1 0" material="color: rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0 1 -1.25" material="color: red"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0 1.6 -1.5" material="color: red"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="-0.9 1 -0.9" material="color: blue"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="-1 1.6 -1" material="color: blue"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0.9 1 -0.9" material="color: green"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="1 1.6 -1" material="color: green"&gt;&lt;/a-box&gt;
        &lt;a-light type="spot" intensity="0.8" color="white" position="1 3 0" angle="60"  
          rotation="-90 0 0" penumbra="0.5" light="castShadow:true;"&gt;&lt;/a-light&gt;
        &lt;a-light type="ambient" intensity="0.5" color="white"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="047" name="sl">
			   <label for="047"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo18a.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Interações com cubos em uma cena.<br><a href="modulo8/exemplo18a.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-135a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-136.png" loading="lazy"/>
  <center><img src="modulo8/objetos/cubos.gif" style="width: 45%; border-radius:5px" loading="lazy"/></center>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-136a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="046bk" name="sl">
			   <label for="046bk"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod199', 'cd199')" onmouseout="outFunc('cd199')"><span class="tooltiptext" id="cd199">Copiar o código</span></button></div>Interação com objetos de uma cena (mãos físicas):
<pre><code id="cod199">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;title&gt;Interações com objetos de uma cena&lt;/title&gt;
    <a alt="referência da biblioteca com suporte de interações">&lt;script src="https://aframe.io/releases/1.1.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de controles para movimentar objetos">&lt;script src="https://unpkg.com/super-hands/dist/super-hands.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de propriedades físicas">&lt;script src="https://rawgit.com/donmccurdy/aframe-physics-system/v4.0.0/dist/aframe-physics-system.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de interação">&lt;script src="https://unpkg.com/aframe-event-set-component@^4.1.1/dist/aframe-event-set-component.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca com propriedades físicas adicionais">&lt;script src="https://unpkg.com/aframe-physics-extras/dist/aframe-physics-extras.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de teleporte">&lt;script src="https://fernandojsg.github.io/aframe-teleport-controls/dist/aframe-teleport-controls.min.js"&gt;&lt;/script&gt;</a>
    &lt;script&gt; <a alt="função de melhoria das propriedades físicas">AFRAME.registerComponent('phase-shift', {</a>
      init: function () {
        var el = this.el
        el.addEventListener('gripdown', function () {
           el.setAttribute('collision-filter', {collisionForces: true})
        })
        el.addEventListener('gripup', function () {
           el.setAttribute('collision-filter', {collisionForces: false})
        })
      }
    });
    &lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene physics shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
           &lt;a-mixin id="cubo" geometry="width: 0.5; height: 0.5; depth: 0.5;" shadow 
             <a alt="propriedades para manipulação dos cubos">hoverable grabbable stretchable draggable droppable</a> event-set__hoveron="_event: hover-start;
             material.opacity: 0.7; transparent: true" event-set__hoveroff="_event: hover-end; 
             material.opacity: 1; transparent: false" <a alt="propriedades físicas dos cubos">dynamic-body=</a>"linearDamping:0.1; 
             angularDamping:0.8; mass:0.5;"&gt;&lt;/a-mixin&gt;
           &lt;a-mixin <a alt="propriedades do piso">id="piso"</a> geometry="width: 10; height: 0.3; depth: 10;" static-body shadow&gt;&lt;/a-mixin&gt;
	   &lt;a-mixin <a alt="propriedades para as mãos físicas usadas para interação">id="mfisicas" physics-collider phase-shift </a> collision-filter="collisionForces: false" 
	     static-body="shape: sphere; sphereRadius: 0.02" super-hands="colliderEvent: collisions; 
	     colliderEventProperty: els; colliderEndEvent: collisions; colliderEndEventProperty: clearedEls;"&gt;&lt;/a-mixin&gt;
        &lt;/a-assets&gt;
        &lt;a-entity <a alt="interações conectadas com a câmera">id="cameraRig"</a>&gt;
           &lt;a-camera id="head" look-controls wasd-controls position="0 1 2" 
             <a alt="interação com o cursor do mouse">cursor="rayOrigin:mouse"</a> static-body="shape: sphere; sphereRadius: 0.001" 
             <a alt="propriedades de colisão e rastreamento">super-hands=</a>"colliderEvent: raycaster-intersection; colliderEventProperty: els;
             colliderEndEvent:raycaster-intersection-cleared; colliderEndEventProperty: clearedEls;"&gt;&lt;/a-camera&gt;
           &lt;a-entity <a alt="teleporte nos cubos e no piso">teleport-controls=</a>"cameraRig: #cameraRig; collisionEntities: [mixin='cubo'],
             [mixin='piso']; teleportOrigin: #head; button: grip;" <a alt="controle com a mão direita">hand-controls="hand: right"</a>
             gearvr-controls daydream-controls&gt;&lt;/a-entity&gt;
           &lt;a-entity <a alt="controle de manipulação da mão esquerda">id="lhand" mixin="mfisicas"</a> hand-controls="hand: left"&gt;&lt;/a-entity&gt;
        &lt;/a-entity&gt;
        <a alt="posições dos objetos na cena">&lt;a-box mixin="piso"</a> class="piso" position="0 -1 0" material="color: rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0 1 -1.25" material="color: red"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0 1.6 -1.5" material="color: red"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="-0.9 1 -0.9" material="color: blue"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="-1 1.6 -1" material="color: blue"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="0.9 1 -0.9" material="color: green"&gt;&lt;/a-box&gt;
        &lt;a-box class="cubo" mixin="cubo" position="1 1.6 -1" material="color: green"&gt;&lt;/a-box&gt;
        &lt;a-light type="spot" intensity="0.8" color="white" position="1 3 0" angle="60"  
          rotation="-90 0 0" penumbra="0.5" light="castShadow:true;"&gt;&lt;/a-light&gt;
        &lt;a-light type="ambient" intensity="0.5" color="white"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="047bk" name="sl">
			   <label for="047bk"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo18a1.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Interações com cubos em uma cena.<br><a href="modulo8/exemplo18a1.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-137.png" loading="lazy"/>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-137a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/cZNOri4vMYM" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-138.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="048" name="sl">
			   <label for="048"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod103', 'cd103')" onmouseout="outFunc('cd103')"><span class="tooltiptext" id="cd103">Copiar o código</span></button></div>Inserção de objeto GLTF na cena:
<pre><code id="cod103">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
            &lt;a-asset-item <a alt="identificação do objeto">id=</a>"objeto" <a alt="caminho do arquivo GLTF">src=</a>"objetos/robo/scene.gltf"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: 0 0 0; minDistance: 0.5; maxDistance: 180; 
          initialPosition: 1 1 2.5"&gt;&lt;/a-entity&gt;
        &lt;a-entity <a alt="posição e escala do objeto na cena">position=</a>"0 0 0.5" scale="0.1 0.1 0.1"&gt;
            &lt;a-gltf-model <a alt="referência do objeto">src=</a>"#objeto" shadow&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-box scale="7 0.1 7" shadow color="rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-light type="ambient" color="#white" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="0.6" position="-1 1 1" 
          light="castShadow:true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="049" name="sl">
			   <label for="049"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo19c.htm" title="Objeto GLTF" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Objeto GLTF inserido na cena.<br><a href="modulo8/exemplo19c.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-138a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="050" name="sl">
			   <label for="050"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod104', 'cd104')" onmouseout="outFunc('cd104')"><span class="tooltiptext" id="cd104">Copiar o código</span></button></div>Inserção de objeto GLTF com animação nativa na cena:
<pre><code id="cod104">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    <a alt="referência da biblioteca para animação nativa de objetos">&lt;script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"&gt;&lt;/script&gt;</a>
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
            &lt;a-asset-item id="objeto" <a alt="caminho do objeto GLTF com animação nativa">src=</a>"objetos/mushroom/scene.gltf"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: 0 0 0; minDistance: 0.05; maxDistance: 180; 
          initialPosition: 1 1 2.5"&gt;&lt;/a-entity&gt;
        &lt;a-entity position="0 0 0.5" scale="0.5 0.5 0.5"&gt;
            &lt;a-gltf-model <a alt="referência do objeto">src=</a>"#objeto" shadow <a alt="habilita a animação nativa do objeto">animation-mixer</a>&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-box scale="3.5 0.1 3.5" position="0 -0.07 0" shadow color="rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-light type="ambient" color="#white" intensity="0.4"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.7" color="white" position="-1 2 1" angle="80" 
          rotation="-90 0 0" penumbra="0.7" light="castShadow:true;"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="051" name="sl">
			   <label for="051"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo19d.htm" title="Objeto GLTF" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Primeiro exemplo de objeto GLTF com animação nativa na cena.<br><a href="modulo8/exemplo19d.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="052" name="sl">
			   <label for="052"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo9/exemplo3aVR.htm" title="Objeto GLTF" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Segundo exemplo de objeto GLTF com animação nativa na cena.<br><a href="modulo9/exemplo3aVR.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-138b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-139.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="053" name="sl">
			   <label for="053"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod105', 'cd105')" onmouseout="outFunc('cd105')"><span class="tooltiptext" id="cd105">Copiar o código</span></button></div>Inserção de objeto GLTF com animação na cena:
<pre><code id="cod105">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
            &lt;a-asset-item id="objeto" <a alt="caminho do arquivo GLTF do avião">src=</a>"objetos/aviao.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item <a alt="caminho do arquivo GLTF da hélice">id=</a>"objeto1" src="objetos/helice.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-mixin id="texto" text="color: black; align: left; width: 8; side:double;"&gt;&lt;/a-mixin&gt;
        &lt;/a-assets&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera="fov:20; near:1;" orbit-controls="target: 0 0 0; minDistance: 0.5; 
          maxDistance: 180; initialPosition: 5 7 21"&gt;&lt;/a-entity&gt;
        &lt;a-entity <a alt="posição e escala dos objetos">position=</a>"0 0 0" scale="0.8 0.8 0.8" rotation="0 -15 0"&gt;
            &lt;a-gltf-model <a alt="objeto estático">src=</a>"#objeto" shadow&gt;&lt;/a-gltf-model&gt;
            &lt;a-entity position="3.54 2.14 0.03" rotation="0 0 13.3"&gt;
                &lt;a-gltf-model <a alt="hélice com animação de rotação">src=</a>"#objeto1" animation="property: rotation; to: 360 0 0; loop: true; 
                  dur: 5000; easing: linear" shadow&gt;&lt;/a-gltf-model&gt;
            &lt;/a-entity&gt;
            &lt;a-text position="0 0.15 3.5" mixin="texto" value="A airplane  |  design: Jeroen Hut" 
              rotation="-45 0 0"&gt;&lt;/a-text&gt;
        &lt;/a-entity&gt;
        &lt;a-box shadow scale="10 0.1 10" position="0 -0.05 0" color="rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-light type="ambient" color="#bbb" intensity="0.6"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="0.8" position="-1 1 1" 
          light="castShadow:true"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="054" name="sl">
			   <label for="054"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo19.htm" title="Objeto GLTF" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Exemplo de objeto GLTF com animação feita com a-frame.<br><a href="modulo8/exemplo19.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-139a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-140.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="055" name="sl">
			   <label for="055"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod106', 'cd106')" onmouseout="outFunc('cd106')"><span class="tooltiptext" id="cd106">Copiar o código</span></button></div>Inserção de objeto GLTF para uso de teleporte:
<pre><code id="cod106">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://fernandojsg.github.io/aframe-teleport-controls/dist/aframe-teleport-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
            &lt;a-asset-item id="objeto" <a alt="objeto GLTF de uma caverna">src=</a>"objetos/cave/scene.gltf"&gt;&lt;/a-asset-item&gt;
            &lt;a-mixin <a alt="piso para usar teleporte">id=</a>"piso" geometry="height:0.1; width:10; depth:20;" shadow&gt;&lt;/a-mixin&gt;
            &lt;img id="ceu" src="imagens/equi4.jpg"&gt;
        &lt;/a-assets&gt;
        &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
        &lt;a-entity id="cameraRig" position="0.5 0 0"&gt;
            &lt;a-camera id="head" wasd-controls look-controls&gt;&lt;/a-camera&gt;
            &lt;a-entity id="rhand" <a alt="propriedades do teleporte">teleport-controls</a>="cameraRig: #cameraRig; collisionEntities: [mixin='piso'];  
              teleportOrigin: #head; button: grip;" hand-controls="hand: right"&gt;&lt;/a-entity&gt;
            &lt;a-entity id="lhand" teleport-controls="cameraRig: #cameraRig; collisionEntities: [mixin='piso']; 
              teleportOrigin: #head; button: grip;" hand-controls="hand: left"&gt;&lt;/a-entity&gt;
        &lt;/a-entity&gt;
        &lt;a-box mixin="piso" position="0 -0.55 0" visible="false"&gt;&lt;/a-box&gt;
        &lt;a-entity position="0 0 0"&gt;
            &lt;a-gltf-model <a alt="referência do objeto na cena">src=</a>"#objeto" shadow material="metallic:0.2; roughness:0.8;"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-light type="ambient" color="#eaeaea" intensity="0.3"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.7" position="1 1 3" light="castShadow:true" 
          rotation="-90 0 0" penumbra="0.5"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="056" name="sl">
			   <label for="056"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo19b.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Primeiro exemplo de objeto GLTF com teleporte.<br><a href="modulo8/exemplo19b.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="057" name="sl">
			   <label for="057"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo19a.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Segundo exemplo de objeto GLTF com teleporte.<br><a href="modulo8/exemplo19a.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-140a.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="058" name="sl">
			   <label for="058"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod107', 'cd107')" onmouseout="outFunc('cd107')"><span class="tooltiptext" id="cd107">Copiar o código</span></button></div>Inserção de objeto OBJ em uma cena:
<pre><code id="cod107">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-orbit-controls@1.3.0/dist/aframe-orbit-controls.min.js"&gt;&lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene reflection shadow="type: pcfsoft" renderer='colorManagement: true;'&gt;
        &lt;a-assets&gt;
            &lt;a-asset-item id="objeto" <a alt="arquivo da geometria OBJ">src=</a>"objetos/bowling.obj"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="objeto-mtl" <a alt="arquivo dos materiais">src=</a>"objetos/bowling.mtl"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;
        &lt;a-sky color="#66ccff"&gt;&lt;/a-sky&gt;
        &lt;a-entity camera orbit-controls="target: 0 0 0; minDistance: 0.05; maxDistance: 180; 
          initialPosition: 1 1 2"&gt;&lt;/a-entity&gt;
        &lt;a-entity position="0 0 0" scale="0.03 0.03 0.03" rotation="-90 0 0"&gt;
            &lt;a-obj-model <a alt="referência do arquivo OBJ">src=</a>"#objeto" <a alt="referência dos materiais">mtl=</a>"#objeto-mtl" shadow &gt;&lt;/a-obj-model&gt;
        &lt;/a-entity&gt;
        &lt;a-box scale="3.5 0.1 3.5" position="0 -0.07 0" shadow color="rgb(100,180,100)"&gt;&lt;/a-box&gt;
        &lt;a-light type="ambient" color="#white" intensity="0.7"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.7" color="white" position="1 2 1" angle="80" rotation="-90 0 0" 
          penumbra="0.7" light="castShadow:true;"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="059" name="sl">
			   <label for="059"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo20.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Primeiro exemplo de objeto GLTF com teleporte.<br><a href="modulo8/exemplo20.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-140b.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-141.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-142.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-143.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-144.png" loading="lazy"/>
  <div class="combo"><details class="sub" style="box-shadow: none;"><summary>&#x1f4c3; Código</summary>
	<p>Veja o código HTML e a renderização da cena.</p>
	  <ul class="slider">
		  <li>
			   <input type="radio" id="060" name="sl">
			   <label for="060"></label>
			   <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod108', 'cd108')" onmouseout="outFunc('cd108')"><span class="tooltiptext" id="cd108">Copiar o código</span></button></div>Cena com objetos GLTF, interação e teleporte:
<pre><code id="cod108">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;title&gt;Pista de Boliche&lt;/title&gt;
    <a alt="referências das bibliotecas com suporte teleporte, interação e física">&lt;script src=</a>"https://aframe.io/releases/1.1.0/aframe.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/super-hands/dist/super-hands.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://rawgit.com/donmccurdy/aframe-physics-system/v4.0.0/dist/aframe-physics-system.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-event-set-component@^4.1.1/dist/aframe-event-set-component.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://unpkg.com/aframe-physics-extras/dist/aframe-physics-extras.min.js"&gt;&lt;/script&gt;
    &lt;script src="https://fernandojsg.github.io/aframe-teleport-controls/dist/aframe-teleport-controls.min.js"&gt;&lt;/script&gt;
    &lt;script&gt; AFRAME.registerComponent('phase-shift', {
    init: function () {
      var el = this.el
        el.addEventListener('gripdown', function () {
          el.setAttribute('collision-filter', {collisionForces: true})
        })
        el.addEventListener('gripup', function () {
          el.setAttribute('collision-filter', {collisionForces: false})
        })
      }
    });
    &lt;/script&gt;
  &lt;/head&gt;
  &lt;body&gt;
    &lt;a-scene shadow="type: pcfsoft"&gt;
        &lt;a-assets&gt;
           &lt;a-mixin <a alt="bolas de boliche com propriedades físicas">id="bola"</a> geometry="radius: 0.3;" material="color: grey; metalness:0.8;" 
             hoverable grabbable stretchable draggable droppable shadow dynamic-body="linearDamping:0.2; 
             angularDamping:0.2; mass:7;" event-set__hoveron="_event: hover-start; material.opacity: 0.7; transparent: true" 
             event-set__hoveroff="_event: hover-end; material.opacity: 1; transparent: false"&gt;&lt;/a-mixin&gt;
           &lt;a-mixin <a alt="pinos com propriedades físicas">id="pino"</a> scale="0.25 0.2 0.25" hoverable grabbable stretchable draggable droppable 
             dynamic-body="linearDamping:0.3; angularDamping:0.3; mass:0.5;" shadow&gt;&lt;/a-mixin&gt;
           &lt;a-asset-item id="bow" <a alt="arquivo GLTF do modelo de pino">src="objetos/bowling.gltf"</a>&gt;&lt;/a-asset-item&gt;
           &lt;a-mixin static-body <a alt="plataforma da pista, com propriedades físicas">id="plataforma"</a> geometry="height:0.1; width:18; depth:1.5;" 
             material="src:#piso2; repeat:15 2; side:double; metalness:0.2; roughness:0.7" shadow&gt;&lt;/a-mixin&gt;
           &lt;a-mixin static-body <a alt="barras laterais da pista">id="lateral"</a> material="src:#piso2; repeat:15 1; side:double; 
             metalness:0.2; roughness:0.7" shadow&gt;&lt;/a-mixin&gt;
           &lt;a-mixin static-body <a alt="piso da pista">id="piso"</a> geometry="height:0.1; width:30; depth:15;" shadow
             material="src:#piso1; repeat:27 14; side:double; metalness:0.2; roughness:0.7"&gt;&lt;/a-mixin&gt;
           <a alt="texturas da cena">&lt;img id=</a>"ceu" src="imagens/bowling.jpg"&gt;
           &lt;img id="piso1" src="imagens/piso_madeira.jpg"&gt;
           &lt;img id="piso2" src="imagens/piso.jpg"&gt;
           &lt;a-mixin <a alt="ponteiro para selecionar os objetos">id="pointer" phase-shift</a> raycaster="showLine: true; objects: .bola, a-sphere, 
	     .pino, a-gltf" collision-filter="collisionForces: false" static-body="shape: sphere; sphereRadius: 0.02" 
	     super-hands="colliderEvent: raycaster-intersection; colliderEventProperty: els; 
	     colliderEndEvent: raycaster-intersection-cleared; colliderEndEventProperty: clearedEls;"&gt;&lt;/a-mixin&gt;
           &lt;a-mixin <a alt="controle para manipular os objetos">id="controle" mixin="pointer"</a> hand-controls="hand: left"&gt;&lt;/a-mixin&gt;
        &lt;/a-assets&gt;
        &lt;a-sky src="#ceu"&gt;&lt;/a-sky&gt;
        &lt;a-entity <a alt="propriedades de interação ligadas à câmera">id="cameraRig"</a> position="3 1.6 2"&gt;
           &lt;a-camera id="head" look-controls wasd-controls position="0 1 2" cursor="rayOrigin:mouse" 
             static-body="shape: sphere; sphereRadius: 0.001" super-hands="colliderEvent: raycaster-intersection; 
             colliderEventProperty: els; colliderEndEvent: raycaster-intersection-cleared; 
             colliderEndEventProperty: clearedEls;"&gt;&lt;/a-camera&gt;
           &lt;a-entity teleport-controls="cameraRig: #cameraRig; collisionEntities: [mixin='piso']; 
             teleportOrigin: #head; button: grip;" hand-controls="hand: right" gearvr-controls daydream-controls&gt;&lt;/a-entity&gt;
           &lt;a-entity id="lhand" mixin="controle" super-hands&gt;&lt;/a-entity&gt;
        &lt;/a-entity&gt;
        &lt;a-entity <a alt="posição do conjunto de objetos da cena">position="0 0 -2"</a>&gt;
           &lt;a-box mixin="piso" position="0 -0.1 0"&gt;&lt;/a-box&gt;
           &lt;a-box mixin="plataforma" position="0 -0.05 0"&gt;&lt;/a-box&gt;
           &lt;a-box mixin="lateral" scale="18.3 0.5 0.1" position="0 0.2 -0.8"&gt;&lt;/a-box&gt;
           &lt;a-box mixin="lateral" scale="18.3 0.5 0.1" position="0 0.2 0.8"&gt;&lt;/a-box&gt;
           &lt;a-entity <a alt="suporte com as bolas de boliche">id="suporte"</a>&gt;
              &lt;a-box mixin="lateral" material="src:#piso2; repeat:2 2;" scale="2.1 0.5 0.1" 
                position="8 0.2 -2.5"&gt;&lt;/a-box&gt;
              &lt;a-box mixin="lateral" material="src:#piso2; repeat:2 2;" scale="2.1 0.5 0.1" 
                position="8 0.2 -1.5" rotation="0 0 0"&gt;&lt;/a-box&gt;
              &lt;a-box mixin="lateral" material="src:#piso2; repeat:2 2;" scale="0.9 0.5 0.1" 
                position="7 0.2 -2" rotation="0 90 0"&gt;&lt;/a-box&gt;
              &lt;a-box mixin="lateral" material="src:#piso2; repeat:2 2;" scale="0.9 0.5 0.1" 
                position="9 0.2 -2" rotation="0 90 0"&gt;&lt;/a-box&gt;
           &lt;/a-entity&gt;
           &lt;a-sphere class="bola" mixin="bola" position="7.5 0 -2"&gt;&lt;/a-sphere&gt;
           &lt;a-sphere class="bola" mixin="bola" position="8 0 -2"&gt;&lt;/a-sphere&gt;
           &lt;a-sphere class="bola" mixin="bola" position="8.5 0 0"&gt;&lt;/a-sphere&gt;
           &lt;a-entity <a alt="posição do conjunto de pinos">position="0 0.7 0.1"</a>&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.6 0 0.15" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.6 0 -0.15" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.6 0 0.45" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.6 0 -0.45" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.3 0 0" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.3 0 -0.3" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8.3 0 0.3" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8 0 0.15" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-8 0 -0.15" mixin="pino"&gt;&lt;/a-gltf-model&gt;
              &lt;a-gltf-model class="pino" src="#bow" position="-7.7 0 0" mixin="pino"&gt;&lt;/a-gltf-model&gt;
           &lt;/a-entity&gt;
        &lt;/a-entity&gt;
        <a alt="iluminação da cena com 1 luz ambiente e 3 spots">&lt;a-light</a> type="spot" intensity="0.7" color="white" position="-8 3 -2" angle="75" 
          rotation="-90 0 0" penumbra="0.3" light="castShadow:true; shadowBias: -0.001;"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.7" color="white" position="0 3 -2" angle="75" 
          rotation="-90 0 0" penumbra="0.3" light="castShadow:true; shadowBias: -0.001;"&gt;&lt;/a-light&gt;
        &lt;a-light type="spot" intensity="0.7" color="white" position="8 3 -2" angle="75"  
          rotation="-90 0 0" penumbra="0.3" light="castShadow:true; shadowBias: -0.001;"&gt;&lt;/a-light&gt;
        &lt;a-light type="ambient" intensity="0.5" color="white"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
  &lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
		   </li>
		   <li>
			   <input type="radio" id="061a" name="sl">
			   <label for="061a"></label>
			   <div class="embed-container"><iframe width="100%" src="modulo8/exemplo21.htm" title="Interações com objetos" frameborder="0" loading="lazy"></iframe></div>
			   <figcaption>Cena com uma pista de boliche: teleporte, objetos GLTF, texturas e interação com os objetos.<br><a href="modulo8/exemplo21.htm" target="_blank">&#x1f517; link da página</a></figcaption>
		   </li>
		</ul>
		<img src="modulo6/0.png" class="fundo" style="visibility:hidden;" loading="lazy"/>
  </details></div>
  <center><img src="modulo8/objetos/boliche.gif" style="width:45%; border-radius:5px;" loading="lazy"/></center>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-144a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
</details>

<details style="border-bottom: 1px solid #a2dec0;">
  <summary id="modulo9">9. Realidade Aumentada</summary>
  <p>Material da página 145 até a página 154.</p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-144.png" loading="lazy"/>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/9Ug6sSLWvf8" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-145.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-146.png" loading="lazy"/>
<div class="combo"><details class="sub"><summary>&#x1f4c3; Código do cabeçalho</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod109', 'cd109')" onmouseout="outFunc('cd109')"><span class="tooltiptext" id="cd109">Copiar o código</span></button></div>Cabeçalho da página em RA de rastreamento de faces
<pre><code id="cod109">&lt;!DOCTYPE html&gt;
&lt;html&gt;
  &lt;head&gt;
    &lt;meta name="viewport" content="width=device-width, <a alt="impede distorções dos objetos">user-scalable=no, minimum-scale=1.0, maximum-scale=1.0"&gt;</a>
    <a alt="referências das bibliotecas para rastreamento de faces">&lt;script src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/dist/mindar-face.prod.js"&gt;&lt;/script&gt;</a>
    <a alt="versão do aframe com suporte">&lt;script src="https://aframe.io/releases/1.2.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    &lt;script src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/dist/mindar-face-aframe.prod.js"&gt;&lt;/script&gt;
    <a alt="função para trocar os objetos GLTF que aparecem na cena">&lt;script&gt;</a>
      document.addEventListener("DOMContentLoaded", function() {
      const list = ["glasses1", "glasses2", "hat1", "hat2", "earring"];
      const visibles = [true, false, false, true, true];
      const setVisible = (button, entities, visible) =&gt; {
        if (visible) {
          button.classList.add("selected");
        } else {
          button.classList.remove("selected");
        }
      entities.forEach((entity) =&gt; {
        entity.setAttribute("visible", visible);
      });
    }
    list.forEach((item, index) =&gt; {
      const button = document.querySelector("#" + item);
      const entities = document.querySelectorAll("." + item + "-entity");
      setVisible(button, entities, visibles[index]);
      button.addEventListener('click', () =&gt; {
        visibles[index] = !visibles[index];
        setVisible(button, entities, visibles[index]);
      });
     });
    })
    &lt;/script&gt;
    <a alt="folha de estilos para posicionar os botões para escolha dos objetos da cena">&lt;style&gt;</a>
      body {margin: 0;}
      .example-container {overflow: hidden; position: absolute; width: 100%; height: 100%;}
      .options-panel {position: fixed; left: 0; top: 0; z-index: 2;}
      .options-panel img {border: solid 2px; width: 50px; height: 50px;	object-fit: cover;  
        cursor: pointer;}
      .options-panel img.selected {border-color: green;}
    &lt;/style&gt;
  &lt;/head&gt;
</code></pre></figcaption>
  </details></div>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-146a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-147.png" loading="lazy"/>
<div class="combo"><details class="sub"><summary>&#x1f4c3; Código do corpo da página</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod110', 'cd110')" onmouseout="outFunc('cd110')"><span class="tooltiptext" id="cd110">Copiar o código</span></button></div>Corpo da página em RA de rastreamento de faces
<pre><code id="cod110">  &lt;body&gt;
    &lt;div class="example-container"&gt;
      <a alt="janela de opções dos objetos da cena">&lt;div class="options-panel"&gt;</a>
        &lt;img id="hat1" src="imagens/hat1.png"&gt;
        &lt;img id="hat2" src="imagens/hat2.png"&gt;
        &lt;img id="glasses1" src="imagens/glasses1.png"&gt;
        &lt;img id="glasses2" src="imagens/glasses2.png"&gt;
        &lt;img id="earring" src="imagens/earring.png"/&gt;
      &lt;/div&gt;
      &lt;a-scene <a alt="cena com referência à função da biblioteca de rastreamento de faces">mindar-face</a> embedded color-space="sRGB" renderer="colorManagement: true, physicallyCorrectLights" vr-mode-ui="enabled: false" device-orientation-permission-ui="enabled: false"&gt;
        <a alt="caminhos dos objetos GLTF da cena">&lt;a-assets&gt;</a>
          &lt;a-asset-item id="headModel" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/sparkar/headOccluder.glb"&gt;&lt;/a-asset-item&gt;
          &lt;a-asset-item id="glassesModel" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/glasses/scene.gltf"&gt;&lt;/a-asset-item&gt;
          &lt;a-asset-item id="glassesModel2" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/glasses2/scene.gltf"&gt;&lt;/a-asset-item&gt;
          &lt;a-asset-item id="hatModel" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/hat/scene.gltf"&gt;&lt;/a-asset-item&gt;
          &lt;a-asset-item id="hatModel2" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/hat2/scene.gltf"&gt;&lt;/a-asset-item&gt;
          &lt;a-asset-item id="earringModel" src="https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.1.4/examples/face-tracking/assets/earring/scene.gltf"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;
        <a alt="configurações da câmera">&lt;a-camera</a> position="0 0 0" active="false"  look-controls-enabled="false" rotation-reader 
          arjs-look-controls="smoothingFactor: 0.05"&gt;&lt;/a-camera&gt;
        <a alt="modelo de uma face, usado para o rastreamento">&lt;a-entity mindar-face-target="anchorIndex: 168"&gt;</a>
            &lt;a-gltf-model mindar-face-occluder position="0 -0.3 0.15" rotation="0 0 0" 
              scale="0.065 0.065 0.065" src="#headModel"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        <a alt="inserção do objeto escolhido, de acordo com o índice selecionado">&lt;a-entity mindar-face-target="anchorIndex: 10"&gt;</a>
            &lt;a-gltf-model rotation="0 -0 0" position="0 1.0 -0.5" 
              scale="0.35 0.35 0.35" src="#hatModel" class="hat1-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-entity mindar-face-target="anchorIndex: 10"&gt;
            &lt;a-gltf-model rotation="0 -0 0" position="0 -0.2 -0.5" 
              scale="0.008 0.008 0.008" src="#hatModel2" class="hat2-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-entity mindar-face-target="anchorIndex: 168"&gt;
            &lt;a-gltf-model rotation="0 -0 0" position="0 0 0" 
              scale="0.01 0.01 0.01" src="#glassesModel" class="glasses1-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-entity mindar-face-target="anchorIndex: 168"&gt;
            &lt;a-gltf-model rotation="0 -90 0" position="0 -0.3 0" 
              scale="0.6 0.6 0.6" src="#glassesModel2" class="glasses2-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-entity mindar-face-target="anchorIndex: 127"&gt;
            &lt;a-gltf-model rotation="-0.1 -0 0" position="0 -0.3 -0.3" 
              scale="0.05 0.05 0.05" src="#earringModel" class="earring-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
        &lt;a-entity mindar-face-target="anchorIndex: 356"&gt;
            &lt;a-gltf-model rotation="0.1 -0 0" position="0 -0.3 -0.3" 
              scale="0.05 0.05 0.05" src="#earringModel" class="earring-entity" visible="false"&gt;&lt;/a-gltf-model&gt;
        &lt;/a-entity&gt;
      &lt;/a-scene&gt;
    &lt;/div&gt;
  &lt;/body&gt;
&lt;/html&gt;  
</code></pre></figcaption>
<p><a href="modulo9/exemplo1.htm" target="_blank">&#x1f517; link da página</a></p>
  </details></div>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-147a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <div class="embed-containerA">
		<iframe width="100%" src="https://www.youtube.com/embed/JLvD8Xt5_kI" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div> 
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-148.png" loading="lazy"/>
<div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod111', 'cd111')" onmouseout="outFunc('cd111')"><span class="tooltiptext" id="cd111">Copiar o código</span></button></div>Código da página em RA baseada em localização
<pre><code id="cod111">&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;
    &lt;meta charset='utf-8'&gt;
    &lt;meta http-equiv='X-UA-Compatible' content='IE=edge'&gt;
    &lt;meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0"&gt;
    <a alt="referência da biblioteca aframe">&lt;script src="https://aframe.io/releases/1.3.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca de RA baseada em localização">&lt;script src="https://unpkg.com/aframe-look-at-component@1.0.0/dist/aframe-look-at-component.min.js"&gt;&lt;/script&gt;</a>
    <a alt="biblioteca da criação da cena em RA">&lt;script src="https://raw.githack.com/AR-js-org/AR.js/master/aframe/build/aframe-ar-nft.js"&gt;&lt;/script&gt;</a>
&lt;/head&gt;

&lt;body style="margin: 0px; overflow: hidden;"&gt;
    &lt;a-scene vr-mode-ui="enabled: false" renderer="logarithmicDepthBuffer: true;" embedded arjs='sourceType: webcam; 
      debugUIEnabled: false;'&gt;
        <a alt="caminhos dos objetos da cena">&lt;a-assets&gt;</a>
            &lt;a-asset-item id="modelo" src="objetos/helicoptero.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo1" src="objetos/helicea.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo2" src="objetos/heliceb.glb"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;

        &lt;a-entity look-at="[gps-camera]" <a alt="insira os valores da latitude e da longitude onde o objeto vai aparecer">gps-entity-place=</a>"latitude: &lt;DIGITE AQUI&gt;; longitude: &lt;DIGITE AQUI&gt;;"&gt;
            <a alt="propriedades do objeto na cena">&lt;a-entity rotation="0 120 0" scale="0.5 0.5 0.5" position="2 0 -4"&gt; </a>
                &lt;a-gltf-model src="#modelo"&gt;&lt;/a-gltf-model&gt;
                &lt;a-entity position="0 0 0"&gt;
                    &lt;a-gltf-model src="#modelo1" <a alt="animação da primeira hélice">animation=</a>"property: rotation; to: 0 360 0; 
                      loop: true; dur: 4000; easing: linear"&gt;&lt;/a-gltf-model&gt;
                &lt;/a-entity&gt;
                &lt;a-entity position="0.037 1.947 5.267" rotation="10 0 0"&gt;
                    &lt;a-gltf-model src="#modelo2" <a alt="animação da segunda hélice">animation=</a>"property: rotation; to: 360 0 0; 
                      loop: true; dur: 3200; easing: linear"&gt;&lt;/a-gltf-model&gt;
                &lt;/a-entity&gt;
            &lt;/a-entity&gt;
        &lt;/a-entity&gt;
		
        <a alt="propriedades da câmera">&lt;a-camera</a> gps-camera rotation-reader active="false" position="0 0 0" 
          look-controls-enabled="false"&gt;&lt;/a-camera&gt;
        <a alt="iluminação da cena">&lt;a-light type=</a>"ambient" color="white" intensity="2"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="1.5" position="-1 1 1"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="1.5" position="1 1 1"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
&lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
<p><a href="modulo9/exemplo2aVR.htm" target="_blank">&#x1f517; link da página do helicóptero em RV</a></p>
  </details></div>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-148a.png" loading="lazy"/>
  <center><img src="modulo9/objetos/ra_local.gif" style="width: 25%; border-radius:5px" loading="lazy"/></center>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-149.png" loading="lazy"/>
	<div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p>Padrões de QR codes 3x3 e 4x4: <a href="https://github.com/artoolkit/ARToolKit5/tree/master/doc/patterns" target="_blank">https://github.com/artoolkit/ARToolKit5/tree/master/doc/patterns</a>
  </p></details></div>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-149a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-150.png" loading="lazy"/>
<div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption><div class="tooltip"><button type="button" onclick="copyEvent('cod112', 'cd112')" onmouseout="outFunc('cd112')"><span class="tooltiptext" id="cd112">Copiar o código</span></button></div>Código da página em RA baseada em marcadores QR codes impressos
<pre><code id="cod112">&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;
    &lt;meta http-equiv="X-UA-Compatible" content="IE=edge" /&gt;
    &lt;meta charset="utf-8" /&gt;
    &lt;meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0"&gt;
    <a alt="referências das bibliotecas aframe e de RA">&lt;script src="https://aframe.io/releases/1.2.0/aframe.min.js"&gt;&lt;/script&gt;</a>
    &lt;script src="https://jeromeetienne.github.io/AR.js/aframe/build/aframe-ar.js"&gt;&lt;/script&gt;
    <a alt="biblioteca de animação nativa de arquivos GLTF">&lt;script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"&gt;&lt;/script&gt;</a>
&lt;/head&gt;

&lt;body style="margin: 0px; overflow: hidden;"&gt;
    &lt;a-scene embedded renderer="logarithmicDepthBuffer: true;" arjs='sourceType: webcam; trackingMethod: best; 
      <a alt="detecção de marcadores 3x3">detectionMode:</a> mono_and_matrix; matrixCodeType: 3x3; debugUIEnabled: false;'&gt;
        <a alt="caminhos dos arquivos dos objetos da cena RA">&lt;a-assets&gt;</a>
            &lt;a-asset-item id="modelo" src="objetos/helicoptero.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo1" src="objetos/helicea.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo2" src="objetos/heliceb.glb"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo3" src="objetos/lamp2/scene.gltf"&gt;&lt;/a-asset-item&gt;
            &lt;a-asset-item id="modelo4" src="objetos/evening/scene.gltf"&gt;&lt;/a-asset-item&gt;
        &lt;/a-assets&gt;

        &lt;a-marker preset="hiro"&gt;
            <a alt="propriedades do objeto que aparecerá sobre o marcador Hiro">&lt;a-entity scale="0.5 0.5 0.5" position="0 0.5 0"&gt;</a>
                &lt;a-gltf-model src="#modelo"&gt;&lt;/a-gltf-model&gt;
                &lt;a-entity position="0 0 0"&gt;
                    &lt;a-gltf-model src="#modelo1" animation="property: rotation; to: 0 360 0; 
                      loop: true; dur: 4000; easing: linear"&gt;&lt;/a-gltf-model&gt;
                &lt;/a-entity&gt;
                &lt;a-entity position="0.037 1.947 5.267" rotation="10 0 0"&gt;
                    &lt;a-gltf-model src="#modelo2" animation="property: rotation; to: 360 0 0; 
                      loop: true; dur: 3200; easing: linear"&gt;&lt;/a-gltf-model&gt;
                &lt;/a-entity&gt;
            &lt;/a-entity&gt;
        &lt;/a-marker&gt;
		
        &lt;a-marker type="barcode" value="20"&gt;
            <a alt="propriedades do objeto que aparecerá sobre o marcador #20">&lt;a-entity scale="0.5 0.5 0.5" position="0 0.05 0"&gt;</a>
                &lt;a-gltf-model src="#modelo3"&gt;&lt;/a-gltf-model&gt;
            &lt;/a-entity&gt;
        &lt;/a-marker&gt;
		
        &lt;a-marker type="barcode" value="17"&gt;
            <a alt="propriedades do objeto que aparecerá sobre o marcador #17">&lt;a-entity position="1.5 0.05 -2"&gt;</a>
                &lt;a-entity position="0 1 0" animation="property:rotation; to:0 360 0; dur:8000; 
                  easing:linear; loop:true"&gt;
                    &lt;a-octahedron color="orange" radius="1" opacity="0.8"&gt;&lt;/a-octahedron&gt;
                    &lt;a-octahedron color="grey" radius="1" wireframe="true"&gt;&lt;/a-octahedron&gt;
                &lt;/a-entity&gt;
                &lt;a-box position="0 1 0" scale="2 2 2" color="blue" opacity="0.6"&gt;&lt;/a-box&gt;
                &lt;a-text position="-1 0 2" rotation="-90 0 0" text="width:6; side:double; color:black"
                  value="Octaedro em RA"&gt;&lt;/a-text&gt;
            &lt;/a-entity&gt;
        &lt;/a-marker&gt;
		
        &lt;a-marker type="barcode" value="10"&gt;
            <a alt="propriedades do objeto que aparecerá sobre o marcador #10">&lt;a-entity scale="0.01 0.01 0.01" position="0 0.05 0"&gt;</a>
                &lt;a-gltf-model src="#modelo4" animation-mixer&gt;&lt;/a-gltf-model&gt;
            &lt;/a-entity&gt;
        &lt;/a-marker&gt;
		
        <a alt="inserção da câmera e da iluminação da cena">&lt;a-entity camera&gt;&lt;/a-entity&gt;</a>
        &lt;a-light type="ambient" color="white" intensity="1"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="1.5" position="-1 1 1"&gt;&lt;/a-light&gt;
        &lt;a-light type="directional" color="white" intensity="1.5" position="1 1 1"&gt;&lt;/a-light&gt;
    &lt;/a-scene&gt;
&lt;/body&gt;
&lt;/html&gt;
</code></pre></figcaption>
<p><a href="modulo9/exemplo3.htm" target="_blank">&#x1f517; link da página</a></p>
  </details></div>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-150a.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-151.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-152.png" loading="lazy"/>
  <center><img src="modulo9/objetos/ra_qr1.gif" style="width: 25%; border-radius:5px;" loading="lazy"/>
  <img src="modulo9/objetos/ra_qr2.gif" style="width: 25%; border-radius:5px; margin-left:20px;" loading="lazy"/>
  </center>
  <br><center><img src="modulo9/objetos/ra_qr3.gif" style="width: 25%; border-radius:5px" loading="lazy"/>
  <img src="modulo9/objetos/ra_qr4.gif" style="width: 25%; border-radius:5px; margin-left:20px;" loading="lazy"/>
  </center>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-153.png" loading="lazy"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
</details>

<h4>página desenvolvida por:</h4> 
<p>Paulo Henrique Siqueira</p>  
<p><b>contato:</b> paulohscwb@gmail.com </p>

<h4>O desenvolvimento deste material faz parte do Grupo de Estudos em Expressão Gráfica (GEEGRAF) da Universidade Federal do Paraná (UFPR)</h4>  

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Licença Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Visualização Científica</span> de <a xmlns:cc="http://creativecommons.org/ns#" href="https://paulohscwb.github.io/visualizacao-cientifica/" property="cc:attributionName" rel="cc:attributionURL">Paulo Henrique Siqueira</a> está licenciado com uma Licença <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Atribuição-NãoComercial-SemDerivações 4.0 Internacional</a>.

<h4>Como citar este trabalho:</h4> 
<p>Siqueira, P.H., "Visualização Científica". Disponível em: <https://paulohscwb.github.io/visualizacao-cientifica/>, Agosto de 2022.</p>
<br><a href="https://doi.org/10.5281/zenodo.8271445"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8271445.svg" alt="DOI"></a>

<h4>Referências:</h4>
<ol>
	<li>A-frame. A web framework for building 3D/AR/VR experiences. Disponível em: <https://aframe.io/>, 2022.</li>
	<li>Anscombe, F. J. Graphs in Statistical Analysis. American Statistician, vol. 27, n. 1, p. 17–21, 1973.</li>
	<li>Card, S. K., Mackinlay, J. D., Shneiderman, B. Readings in Information Visualization Using Vision to Think. San Francisco: Browse books, 1999.</li>
	<li>Eler, D. M. Visualização de Informação. Disponível em: <https://daniloeler.github.io/teaching/VISUALIZACAO>, 2020.</li>
	<li>Horst, A. M., Hill, A. P., Gorman, K. B. Palmerpenguins: Palmer Archipelago (Antarctica) penguin data. Disponível em: <https://allisonhorst.github.io/palmerpenguins/>. doi: 10.5281/zenodo.3960218, 2020.</li>
	<li>Keim, D. A. Information Visualization and Visual Data Mining. IEEE Transactions on Visualization and Computer Graphics, vol. 8, n. 1, p. 1–8, 2002.</li>
	<li>Keller, P. R, Keller, M. M. Visual Cues: Pratical Data Visualization. Los Alamitos, CA: IEEE Computer Society Press, 1994.</li>
	<li>Moro, C. et al. The effectiveness of virtual and augmented reality in health sciences and medical anatomy. Anatomical sciences education, v. 10, n. 6, p. 549–559, 2017.</li>
	<li>Siqueira, P. H. Desenvolvimento de ambientes web em Realidade Aumentada e Realidade Virtual para estudos de superfícies topográficas. Revista Brasileira de Expressão Gráfica, v. 7, n. 2, p. 21–44, 2019.</li>
	<li>Shneiderman, B. The eyes have it: a task by data type taxonomy for information visualization. In: Proceedings of the 1996, IEEE Symposium on Visual Languages, p. 336–343. Washington, DC: IEEE Computer Society, 1996.</li>
	<li>Telea, A. C. Data visualization: principles and practice. Boca Raton: CRC Press, 2015.</li>
	<li>Ward, M., Grinstein, G.G., Keim, D. Interactive data visualization foundations, techniques, and applications. Massachusetts: A K Peters, 2010.</li>
	<li>Williams, J. G., Sochats, K. M., Morse, E. Visualization. Annual Review of Information Science and Technology (ARIST), v. 30, p. 161–207, 1995.</li>
<ol>
<script>
    function copyEvent(id, id1)
    {
        var str = document.getElementById(id);
        window.getSelection().selectAllChildren(str);
        document.execCommand("Copy")
		window.getSelection().collapseToStart();
		var tooltip = document.getElementById(id1);
		tooltip.innerHTML = "Código copiado!";
    }
	function outFunc(id) {
		var tooltip = document.getElementById(id);
		tooltip.innerHTML = "Copiar o código";
	}
</script>
