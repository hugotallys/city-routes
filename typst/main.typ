#import "@preview/rubber-article:0.3.1": *
#import "@preview/lovelace:0.3.0": *

#let title = "Título Aqui"
#let authors = ("Hugo Oliveira", "Matheus Batista")

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

// Medium bold table header.
#show table.cell.where(y: 0): set text(weight: "medium")

// Bold titles.
#show table.cell.where(x: 1): set text(weight: "bold")

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

= Modelagem

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

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  table.header[Month][Title][Author][Genre],
  
  [January], [The Great Gatsby], [F. Scott Fitzgerald], [Classic],
  [February], [To Kill a Mockingbird], [Harper Lee], [Drama],
  [March], [1984], [George Orwell], [Dystopian],
  [April], [The Catcher in the Rye], [J.D. Salinger], [Coming-of-Age],
)

= Conclusão 

#pagebreak()
#show: appendix.with(
  title: "Anexos",
)

= Anexo 1
#lorem(35)
