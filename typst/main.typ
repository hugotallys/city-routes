#import "@preview/rubber-article:0.3.1": *
#import "@preview/lovelace:0.3.0": *

#let title = "Título Aqui"
#let authors = ("Hugo Oliveira", "Matheus Batista")

// Paragraph spacing and identation

#set par(leading: 0.75em, spacing: 1.0em)
#set par(first-line-indent: (
  amount: 1.5em,
  all: true,
))
#set block(spacing: 2em)

// Article Header and Title

#show: article.with(
  lang: "pt",
  show-header: true,
  header-titel: title,
  eq-numbering: "(1.1)",
  eq-chapterwise: true,
)

#maketitle(
  title: title,
  authors: authors,
  // date: datetime.today().display("[day] [month repr:long] [year]"),
  date: datetime.today().display("[day padding:none] abril [year]")
)

// Table Styling

// Bold table header.
#show table.cell.where(y: 0): set text(weight: "bold")

// See the strokes section for details on this!
#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

#set table(
  fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },
  stroke: frame(rgb("21222C")),
)

// Document Content
// Some example content has been added for you to see how the template looks like.

= Introdução

@glacier shows a glacier. Glaciers are complex systems.

#figure(
  image("glacier.jpg", width: 50%),
  caption: [A curious figure.],
) <glacier>

A @peasTable contém a modelagem PEAS do problema.

= Modelagem

O objetivo inicial da resolução do problema da rota de menor custo é a construção de uma rede de cidades conectadas por estradas. Tal rede é representada por um grafo $G = (C, E)$, onde $C = {C_1, C_2, dots, C_N}$ é o conjunto de $N$ cidades possíveis de serem visitadas e $E$ o conjunto de estradas que ligam tais cidades. Dadas duas cidades $C_i$ e $C_j$ quaisquer, diremos que existe uma estrada ligando ambas as cidades ($(C_i, C_j) in E$) quando a distância entre elas for menor ou igual a um limiar $r$:

$
  d(C_i, C_j) <= r
$

Para efeitos práticos, iremos considerar a função de distância $d$ entre duas cidades como a _distância de Harvesine_ entre suas coordenadas geográficas (latitude e longitude). Mais detalhes sobre como o cálculo dessa distância é feito pode ser consultado na @harvesine do apêndice.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr), align: left,
    table.header[Agente][Medida de Performance][Ambiente][Atuadores][Sensores],
    "Sistema de\nPlanejamento de Rota", "Rota de menor distância,\nvisitar cidades menos populosas", "Estradas, \ncidades", "Escolha da cidade de destino em cada etapa do caminho", "Cidades vizinhas da cidade atual,\ndistância entre cidades e\npopulação das cidades",
  ),
  caption: [Modelagem *PEAS* do sistema de planejamento de rota.],
) <peasTable>

#lorem(20)
$
x_(1,2) = (-b plus.minus sqrt(b^2 - 4 a c)) / (2 a)
$
#lorem(20)

= Implementação

#figure(
  kind: "algorithm",
  supplement: [Algorithm],

  pseudocode-list(booktabs: true, numbered-title: [My cool algorithm])[
    + $l <-- A$ do something
    + do something else
    + *while* still something to do
      + do even more
      + *if* not done yet *then*
        + wait a bit
        + resume working
      + *else*
        + go home
      + *end*
    + *end*
  ]
) <cool>

See @cool for details on how to do something cool.

= Resultados

= Conclusão 

#pagebreak()
#show: appendix.with(
  title: "Apêncice",
)

= Cáclulo da Distância de Harvesine <harvesine>
#lorem(35)
