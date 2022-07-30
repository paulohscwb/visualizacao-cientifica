<link rel="stylesheet" href="imagens/style.css">

<h2 id="inicio">Códigos, algoritmos, exemplos e aplicações</h2>

<p>Esta página contém os códigos, algoritmos e exemplos das técnicas mostradas na disciplina de Visualização Científica.</p>
<p>A apostila está disponível no link: <a href="imagens/apostila_2022.pdf" target="_blank">apostila de Visualização Científica</a></p>

<details>
  <summary id="modulo1">1. Introdução</summary>
  <p>Material da página 1 até a página 14.</p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-0.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-1.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-2.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-3.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-4.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-5.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-6.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-7.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-8.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-9.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-10.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-11.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-12.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
   <img src="modulo1/59f0152f9f78561f6fb413c7e4f88ba0-13.png"/>
   <p class="topop"><a href="#modulo1" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo2">2. Conceitos básicos e estruturais de visualização</summary>
  <p>Material da página 14 até a página 24.</p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-13.png"/>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Código em C++ com Matplotlib:
<pre><code>#include <a alt="vetores de coordenadas">&lt;vector&gt;</a> 
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
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Código em Python com Matplotlib:
<pre><code>import <a alt="gráfico plt da biblioteca matplotlib ">matplotlib.pyplot as plt</a> 

<a alt="coordenadas x">x =</a> [0, 1, 2, 3, 4, 5]
<a alt="coordenadas y">y =</a> [1, 4, 9, 16, 32, 64]

<a alt="gráfico de dispersão 2D">plt.scatter</a>(x, y, <a alt="marcador vermelho">color =</a> 'red', <a alt="marcador circular">marker =</a> 'o')</a>
<a alt="comando para visualizar o gráfico">plt.show()</a>

</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-14b.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-15.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de dispersão 3D:
<pre><code>import <a alt="gráfico plt da biblioteca matplotlib ">matplotlib.pyplot as plt</a> 

<a alt="coordenadas x">x =</a> [0, 1, 2, 3, 4, 5]
<a alt="coordenadas y">y =</a> [1, 4, 9, 16, 32, 64]
<a alt="coordenadas z">z =</a> [2, 7, 11, 5, 3, 1]

<a alt="tipo de projeção 3D; gráfico atribuído na variável ax">ax =</a> plt.figure().add_subplot(projection = '3d')

<a alt="gráfico de dispersão 3D, marcador circular e vermelho">ax.scatter(x, y,</a> z, color = 'r', marker = 'o')

plt.show()

</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-15a.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-16.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-17.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-18.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de dispersão 2D com rótulos:
<pre><code>import matplotlib.pyplot as plt 

x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 9, 16, 32, 64, 128]
<a alt="rótulos dos pontos">rotulos =</a> ['A', 'B', 'C', 'D', 'E', 'F', 'G']

<a alt="laço para rotular cada ponto">for i, txt in enumerate(rotulos):</a>
    plt.annotate(txt, (x[i], y[i]))
	
plt.plot(x, y, <a alt="marcador laranja">color =</a> 'orange', <a alt="marcador triângular">marker =</a> '^', <a alt="linha contínua">linestyle =</a> '-')</a>

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-18a.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-19.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de dispersão 3D com rótulos:
<pre><code>import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection = '3d')

x = [0, 1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 32, 64]
z = [2, 7, 11, 5, 3, 1]
<a alt="rótulos dos pontos">rotulos =</a> ['A', 'B', 'C', 'D', 'E', 'F']

<a alt="gráfico de dispersão 3D">ax.scatter(x, y,</a> z, color = 'r', marker = 'o')</a>

<a alt="laço para rotular cada ponto">for x, y, z, tag in zip(x, y, z, rotulos):</a>
    label = tag
    ax.text3D(x, y, z, label, <a alt="direção dos rótulos: eixo z">zdir = 'z'</a>)
	
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-19a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de curvas 2D com legendas:
<pre><code>import matplotlib.pyplot as plt
import <a alt="biblioteca de operações matemáticas">numpy as np</a>

<a alt="intervalo [0, 5] com espaçamento 0.1">x = np.arange(0, 5, 0.1)</a>

<a alt="função linear">plt.plot</a>(x, x, <a alt="linha tracejada azul">'b--',</a> label = 'y = x')</a>
<a alt="função linear, linha contínua verde">plt.plot</a>(x, 2*x+1, <a alt="linha contínua verde">'g-',</a> label = 'y = 2x + 1')</a>
<a alt="função quadrática">plt.plot</a>(x, x**2+2*x+3, <a alt="linha traço-ponto vermelha">'r-.',</a> label = 'y = x^2 + 2x + 3')</a>

<a alt="rótulo do eixo x">plt.xlabel('x')</a>
<a alt="rótulo do eixo y">plt.ylabel('y')</a>
<a alt="título do gráfico">plt.title('Gráfico de curvas 2D')</a>

plt.show()
plt.legend()
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-20.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráficos de curvas 2D:
<pre><code>import matplotlib.pyplot as plt
import numpy as np

<a alt="definição da função f">def f(x):</a>
    return np.exp(-x) * np.cos(2*np.pi*x)

<a alt="intervalo do primeiro gráfico">x1 = np.arange(5, 12, 0.05)</a>
<a alt="intervalo do segundo gráfico">x2 = np.arange(-2, 5, 0.05)</a>

plt.figure()
<a alt="1 linha e 2 colunas de gráficos: gráfico 121">plt.subplot(121)</a>
plt.plot(x1, f(x1), <a alt="linha tracejada azul no intervalo x1">'b--',</a> x2, f(x2), <a alt="traço e ponto verde no intervalo x2">'g-.')</a>

<a alt="1 linha e 2 colunas de gráficos: gráfico 122">plt.subplot(122)</a>
plt.plot(x2, np.cos(2*np.pi*x2), <a alt="linha pontilhada laranja no intervalo x2">color = 'orange', linestyle = ':')</a>
plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-20a.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-21.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráficos de curvas 3D:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-22.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico da hélice cilíndrica:
<pre><code>import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection = '3d')

d = 5
<a alt="grid para a variável z">z =</a> np.linspace(-10, 10, 100)</a>
<a alt="equação com parâmetro d para x">x =</a> d * np.sin(z)
<a alt="equação com parâmetro d para y">y =</a> d * np.cos(z)

<a alt="gráfico da hélice cilíndrica com linha contínua verde">ax.plot(x, y, z, 'g-', label = 'hélice cilíndrica')</a>
ax.legend()

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-22a.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-23.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico da hélice cilíndrica com segmentos projetantes:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo2/59f0152f9f78561f6fb413c7e4f88ba0-23a.png"/>
  <p class="topop"><a href="#modulo2" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo3">3. Fundamentos dos dados</summary>
  <p>Material da página 24 até a página 54.</p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-23.png"/>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-24.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-25.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Links</summary>
  <figcaption>Conjunto Iris: <a href="https://archive.ics.uci.edu/ml/datasets/iris" target="_blank">https://archive.ics.uci.edu/ml/datasets/iris</a>
  <br>Conjunto dos Pinguins: <a href="https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html" target="_blank">https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html</a>
  <br>Outros conjuntos de dados: <a href="https://www.maptive.com/free-data-visualization-data-sets/" target="_blank">https://www.maptive.com/free-data-visualization-data-sets/</a>
  </figcaption></details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-25a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-26.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com matplotlib (cor e movimento):
<pre><code><a alt="biblioteca para leitura dos dados em formato CSV">import pandas as pd</a>
import numpy as np
from matplotlib import pyplot as plt

iris = <a alt="leitura do arquivo CSV">pd.read_csv('C:/dados/iris.csv')</a>

<a alt="variável usada para contar o número de registros">incl = np.array(iris.loc[:,'Largura da Sépala'])</a>
<a alt="variável x">x = np.array(iris.loc[:,'Comprimento da Sépala'])</a>
<a alt="variável y">y = np.array(iris.loc[:,'Comprimento da Pétala'])</a>
<a alt="atributo usado para separação dos dados">z = np.array(iris.loc[:,'Espécie'])</a>

<a alt="criação da grade com a quantidade de registros">i = np.arange(0, len(incl), 1)</a>
<a alt="ajuste dos dados usando os limites max e min">j = (incl - min(incl))/(max(incl) - min(incl))</a>
<a alt="amplitude máxima de 45&deg;">w = -45*j</a>

<a alt="rótulos dos dados com as espécies">label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']</a>
<a alt="vetor com a cor de cada espécie">cor = ['orange', 'green', 'red']</a>
j = 0

<a alt="laço usado para separar as espécies">for k in i:</a>
    <a alt="marcador de cada registro">marker = (2, 1, w[k])</a>
    if <a alt="com os dados ordenados, o marcador muda quando z[k] &ne; z[k-1]">z[k] == z[k-1]:</a>
        plt.plot(x[k], y[k], <a alt="marcadores">marker =</a> marker, <a alt="tamanho dos marcadores">markersize =</a> 10, <a alt="cores dos marcadores">color =</a> cor[j - 1], <a alt="taxa de opacidade">alpha =</a> 0.6)
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-26a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-27.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com matplotlib (textura):
<pre><code>import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

iris = pd.read_csv('C:/dados/iris.csv')

x = np.array(iris.loc[:,'Comprimento da Sépala'])
y = np.array(iris.loc[:,'Comprimento da Pétala'])
z = np.array(iris.loc[:,'Espécie'])

i = np.arange(0, len(x), 1)

label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
<a alt="textura de cada espécie">tex = ['\\', '.', '+']</a>
<a alt="marcadores de cada espécie">marker = ['^', 'X', 'o']</a>
j = 0

<a alt="laço usado para separar as espécies">for k in i:</a>
    if <a alt="com os dados ordenados, o marcador muda quando z[k] &ne; z[k-1]">z[k] == z[k-1]:</a>
        plt.scatter(x[k], y[k], <a alt="marcadores">marker = marker[j - 1]</a>, <a alt="tamanho dos marcadores">s =</a> 200, <a alt="cor dos marcadores">facecolor =</a> 'white', 
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-27a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-28.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com Seaborn (cores):
<pre><code>import pandas as pd
<a alt="biblioteca Seaborn">import seaborn as sns</a>

iris = pd.read_csv('C:/dados/iris.csv')

sns.relplot(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    <a alt="separação de classes: atributo Espécie">hue =</a> 'Espécie', <a alt="marcadores triangulares">marker =</a> '>', <a alt="paleta de cores em tons de azul">palette =</a> 'Blues')</a>
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-28a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-29.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-30.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-31.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-32.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com Seaborn (cores e tamanhos):
<pre><code>import pandas as pd
import seaborn as sns

iris = pd.read_csv('C:/dados/iris.csv')

sns.relplot(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    <a alt="separação de classes: atributo Espécie">hue =</a> 'Espécie', <a alt="marcadores quadrados">marker =</a> 's', <a alt="paleta de cores em tons de vermelho">palette =</a> 'Reds', <a alt="tamanhos dos marcadores">size =</a> 'Largura da Sépala')
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-32a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-33.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-34.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-37.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-38.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-39.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-40.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-43.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-45.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-46.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-47.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-48.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-49.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-50.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-51.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-52.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-53.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo4">4. Taxonomia dos dados</summary>
  <p>Material da página 54 até a página 81.</p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-53.png"/>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-54.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-55.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-56.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-57.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-58.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-59.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-60.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-61.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-62.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-63.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-64.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-65.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-66.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-67.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-68.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-69.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-70.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-71.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-72.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-73.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-75.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-76.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-78.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-79.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-80.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo5">5. Linhas, polígonos, poliedros e superfícies</summary>
  <p>Material da página 81 até a página 92.</p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-80.png"/>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-83.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-84.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-85.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-87.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-88.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-89.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-90.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-91.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo6">6. Modelos de iluminação</summary>
  <p>Material da página 92 até a página 105.</p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-91.png"/>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-92.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-93.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-94.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-95.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-96.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-97.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-98.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-99.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-100.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-101.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-102.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-103.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
  <img src="modulo6/59f0152f9f78561f6fb413c7e4f88ba0-104.png"/>
  <p class="topop"><a href="#modulo6" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo7">7. Câmera</summary>
  <p>Material da página 105 até a página 114.</p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-104.png"/>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-105.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-106.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-107.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-108.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-109.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-110.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-111.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-112.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
  <img src="modulo7/59f0152f9f78561f6fb413c7e4f88ba0-113.png"/>
  <p class="topop"><a href="#modulo7" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo8">8. Realidade Virtual</summary>
  <p>Material da página 115 até a página 144.</p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-114.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-115.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-116.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-117.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-118.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-119.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-120.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-121.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-122.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-123.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-124.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-125.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-126.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-127.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-128.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-129.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-130.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-131.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-132.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-133.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-134.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-135.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-136.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-137.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-138.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-139.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-140.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-141.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-142.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
  <img src="modulo8/59f0152f9f78561f6fb413c7e4f88ba0-143.png"/>
  <p class="topop"><a href="#modulo8" class="topo">voltar ao topo</a></p>
</details>

<details style="border-bottom: 1px solid #a2dec0;">
  <summary id="modulo9">9. Realidade Aumentada</summary>
  <p>Material da página 144 até a página 152.</p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-143.png"/>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-144.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-145.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-146.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-147.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-148.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-149.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-150.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
  <img src="modulo9/59f0152f9f78561f6fb413c7e4f88ba0-151.png"/>
  <p class="topop"><a href="#modulo9" class="topo">voltar ao topo</a></p>
</details>

<h4>página desenvolvida por:</h4> 
<p>Paulo Henrique Siqueira</p>  
<p><b>contato:</b> paulohscwb@gmail.com </p>

<h4>O desenvolvimento deste material faz parte do Grupo de Estudos em Expressão Gráfica (GEEGRAF) da Universidade Federal do Paraná (UFPR)</h4>  

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Licença Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Visualização Científica</span> de <a xmlns:cc="http://creativecommons.org/ns#" href="https://paulohscwb.github.io/visualizacao-cientifica/" property="cc:attributionName" rel="cc:attributionURL">Paulo Henrique Siqueira</a> está licenciado com uma Licença <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Atribuição-NãoComercial-SemDerivações 4.0 Internacional</a>.

<h4>Como citar este trabalho:</h4> 
<p>Siqueira, P.H., "Visualização Científica". Disponível em: <https://paulohscwb.github.io/visualizacao-cientifica/>, Agosto de 2022.</p>

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
