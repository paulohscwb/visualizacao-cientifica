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
<a alt="função linear">plt.plot</a>(x, 2*x+1, <a alt="linha contínua verde">'g-',</a> label = 'y = 2x + 1')</a>
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
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com Seaborn (dispersão e frequência):
<pre><code>import pandas as pd
import seaborn as sns

iris = pd.read_csv('C:/dados/iris.csv')

<a alt="mostra a grade">sns.set_style("whitegrid")</a>
<a alt="função de dispersão e frequência">sns.jointplot</a>(data = iris, x = 'Comprimento da Sépala', y = 'Comprimento da Pétala',
    hue = 'Espécie', <a alt="marcadores circulares">marker =</a> 'o', <a alt="paleta de cores rainbow">palette =</a> 'rainbow')
	
</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-33a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-34.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados dos pinguins com Seaborn (regressão linear):
<pre><code>import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
<a alt="função de dispersão e frequência">sns.lmplot</a>(data = pinguins, x = 'Comprimento do bico', y = 'Massa corporal',
    hue = 'Espécie', <a alt="paleta de cores rocket">palette =</a> 'rocket')

</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-34a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code>import pandas as pd
import seaborn as sns

pinguins = pd.read_csv('C:/dados/penguin2.csv')

sns.set_style("whitegrid")
<a alt="dados que devem ser desconsiderados">pinguins.drop(['Id','Ano'],</a> inplace = True, axis = 1)
<a alt="combinação de gráficos">sns.pairplot</a>(data = pinguins, hue = 'Espécie', <a alt="paleta de cores cubehelix">palette =</a> 'cubehelix')

</code></pre></figcaption>
  </details></div>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code>import pandas as pd
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-35b.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados dos pinguins com Seaborn (combinações de gráficos):
<pre><code>import pandas as pd
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-36a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Conjunto de dados Iris com Plotly (dispersão 3D):
<pre><code>import pandas as pd
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-37.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-38.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-39.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Duas hélices (movimento):
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-40.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 2D:
<pre><code>import matplotlib.pyplot as plt

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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 2D:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-41b.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 2D:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 3D:
<pre><code>import matplotlib.pyplot as plt

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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-42b.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-43.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 3D:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-43a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 3D:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-44a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-45.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Dados vetoriais 3D:
<pre><code>import numpy as np
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-45a.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-46.png"/>
  <p class="topop"><a href="#modulo3" class="topo">voltar ao topo</a></p>
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-47.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Linhas de fluxo 2D:
<pre><code>import plotly.figure_factory as ff
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
  <img src="modulo3/59f0152f9f78561f6fb413c7e4f88ba0-47a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Linhas de fluxo 3D:
<pre><code>import plotly.graph_objects as go
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
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico radar (polar):
<pre><code>import plotly.io as pio
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-63a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-64.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico com coordenadas paralelas:
<pre><code>import pandas as pd
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-64a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-65.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico com seleção interativa:
<pre><code><a alt="biblioteca bokeh de gráficos interativos">from bokeh.layouts import gridplot</a>
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-65a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-66.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-67.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Grafo orientado:
<pre><code><a alt="biblioteca networkx de grafos orientados">import networkx as nx</a>
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-67a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-68.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Grafo orientado para circuito Hamiltoniano:
<pre><code>import networkx as nx
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-68a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-69.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Grafo orientado para o problema do Caixeiro Viajante:
<pre><code>import networkx as nx
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-69a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-70.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-71.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de setores:
<pre><code>import plotly.io as pio
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-71a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-72.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Treeview:
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-72a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-73.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Treeview:
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-73a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Treeview:
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Sunburst (aneis aninhados):
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74b.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Sunburst (aneis aninhados):
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-74c.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico Sunburst (aneis aninhados):
<pre><code>import plotly.express as px
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-75.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-76.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico RadViz:
<pre><code>import pandas as pd
from matplotlib import pyplot as plt

pinguin = pd.read_csv('C:/dados/penguin2.csv', header = 0, <a alt="colunas que contém os dados que serão usados">usecols =</a> [1,3,4,5,6,8])

ax = plt.grid(color = '#d5f8e3', linewidth = 0.5)
fig = <a alt="função para criar o gráfico RadViz">pd.plotting.radviz</a>(pinguin, <a alt="critério de separação dos dados">'Espécie'</a>, colormap = 'rainbow', alpha = 0.6, ax = ax)

fig.show
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de enxame (swarm):
<pre><code>import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico swarm">sns.swarmplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', <a alt="critério de separação">hue = 'Sexo'</a>, data = pinguin)
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-77b.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de enxame (swarm) com diagrama em caixas (boxplot):
<pre><code>import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico boxplot">sns.boxplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', data = pinguin)
sns.swarmplot(x = 'Comprimento da nadadeira', y = 'Espécie', hue = 'Sexo', data = pinguin)
</code></pre></figcaption>
  </details></div>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-78.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Gráfico de enxame (swarm) com violino:
<pre><code>import pandas as pd
import seaborn as sns

pinguin = pd.read_csv('C:/dados/penguin2.csv')
<a alt="função para criar o gráfico violino">sns.violinplot</a>(x = 'Comprimento da nadadeira', y = 'Espécie', data = pinguin, 
    palette = 'Oranges')
sns.swarmplot(x = 'Comprimento da nadadeira', y = 'Espécie', hue = 'Sexo', data = pinguin,  
    palette = 'Blues')
</code></pre></figcaption>
  </details></div>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-78a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-79.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Reconhecimento de imagens:
<pre><code><a alt="biblioteca para carregar imagens">from PIL import Image</a>
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
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-79a.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
  <img src="modulo4/59f0152f9f78561f6fb413c7e4f88ba0-80.png"/>
  <p class="topop"><a href="#modulo4" class="topo">voltar ao topo</a></p>
</details>

<details>
  <summary id="modulo5">5. Linhas, polígonos, poliedros e superfícies</summary>
  <p>Material da página 81 até a página 92.</p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-80.png"/>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um retângulo:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de uma elipse:
<pre><code>import matplotlib.pyplot as plt
<a alt="biblioteca de elipses">from matplotlib.patches import Ellipse</a>

fig, ax = plt.subplots()
patch = <a alt="função para desenhar a elipse">Ellipse</a>((0.5, 0.5), <a alt="diâmetro maior">0.7</a>, <a alt="diâmetro menor">0.3</a>, color = 'orange')
ax.add_patch(patch)

plt.show()
</code></pre></figcaption>
  </details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-81b.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção da superfície lateral de um cilindro circular reto:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um cilindro circular reto:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-82b.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-83.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um cone circular reto:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-83a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um cone circular reto (coordenadas polares):
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-84.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-85.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de uma superfície:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-85a.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um poliedro:
<pre><code>from matplotlib import pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de um poliedro com rótulos dos vértices:
<pre><code>from matplotlib import pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-86b.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-87.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de uma superfície com triangulação:
<pre><code>import matplotlib.pyplot as plt
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-87a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Construção de uma superfície com coordenadas de um arquivo:
<pre><code>import numpy as np
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-88.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-89.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Triangulação de um objeto 3D de extensão PLY:
<pre><code><a alt="biblioteca para leitura de arquivo PLY">from plyfile import PlyData</a>
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-89a.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Triangulação de um objeto 3D de extensão PLY:
<pre><code>from plyfile import PlyData
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-90.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f517; Link</summary>
  <p><a href="https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html" target="_blank">https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html</a></p></details></div>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-90a.png"/>
  <p class="topop"><a href="#modulo5" class="topo">voltar ao topo</a></p>
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-91.png"/>
  <div class="combo"><details class="sub"><summary>&#x1f4c3; Código</summary>
  <figcaption>Triangulação de um objeto 3D de extensão PLY:
<pre><code>import numpy as np
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
  <img src="modulo5/59f0152f9f78561f6fb413c7e4f88ba0-91a.png"/>
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
