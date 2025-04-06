#import "@preview/rubber-article:0.3.1": *
#import "@preview/lovelace:0.3.0": *

#show: article.with(
  show-header: true,
  header-titel: "Solving Problems by Searching",
  eq-numbering: "(1.1)",
  eq-chapterwise: true,
)

#maketitle(
  title: "Solving Problems by Searching",
  authors: ("Hugo Oliveira", "Matheus Batista"),
  date: datetime.today().display("[day]. [month repr:long] [year]"),
)

// Some example content has been added for you to see how the template looks like.
= Introduction
#lorem(60)

#figure(
  rect(width: 4cm, height: 3cm),
  caption: [#lorem(30)],
)

== In this paper
#lorem(20)
$
x_(1,2) = (-b plus.minus sqrt(b^2 - 4 a c))/ (2 a)
$
#lorem(20)

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

=== Contributions
#lorem(40)

= Related Work
#lorem(300)

$
y = k x + d
$
#lorem(50)

#show: appendix.with(
  title: "Appendix",
)

= Appendix 1
#lorem(35)

== Some more details
#lorem(20)


