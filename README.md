# GSDMM: Short text clustering

This project implements the Gibbs sampling algorithm for a Dirichlet Mixture Model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for the 
clustering of short text documents. 
Some advantages of this algorithm:
 - It requires only an upper bound `K` on the number of clusters
 - With good parameter selection, the model converges quickly
 - Space efficient and scalable

## The Movie Group Process

In their paper, the authors introduce a simple conceptual model for explaining the GSDMM called the Movie Group Process.

Imagine a professor is leading a film class. At the start of the class, the students
are randomly assigned to `K` tables. Before class begins, the students make lists of
their favorite films. The professor repeatedly reads the class role. Each time the student's name is called,
the student must select a new table satisfying one or both of the following conditions:

- The new table has more students than the current table.
- The new table has students with similar lists of favorite movies.

By following these steps consistently, we might expect that the students eventually arrive at an "optimal" table configuration.

## Usage

### install
```
pip install git+https://github.com/RaffaeleMorganti/gsdmm.git
```

### use
```python
from gsdmm import GSDMM
gsdmm = GSDMM()

texts = [
        "where the red dog lives",
        "red dog lives in the house",
        "blue cat eats mice",
        "monkeys hate cat but love trees",
        "green cat eats mice",
        "orange elephant must forget",
        "monkeys eat banana",
        "monkeys live in trees"
    ]

clust = gsdmm.fit(texts)
```

### other
```python
prob = gsdmm.predict_proba(texts)
pred = gsdmm.predict(texts)
imp1 = gsdmm.get_importances()
imp2 = gsdmm.get_avg_importances()
info = gsdmm.get_clust_info()
pars = gsdmm.get_params()
plot = gsdmm.get_wordclouds(imp1)
```
