# The geoML manual

Theory, and how it ties to the code. Each chapter teaches one piece of the
model in geostatistical terms, ends its theory sections with an **In the
code** box naming the classes that implement them, and carries small runnable
examples on the bundled data. The full case studies close the manual.

Derivations live in the papers. They are cited in the text and listed in
full in a **References** section at the end of each chapter that has any.
The design documents under `docs/` carry the engineering decisions and
measurements, and are linked where they apply.

Code follows the package's own conventions, which the `geo-ml` skill in
`plugins/geoml/` records: variables and attributes are reached by **tree
path** (`container.values("V/prediction")`), never by attribute chain.

> **Written by Claude.** Every code block
> here runs, and the numbers quoted in the text were measured rather than
> recalled, but that guarantees only that the code executes and that the
> figures match what produced them. There may be errors in the geostatistical reasoning, 
> in the emphasis, in
> the choice of what deserved a chapter, and in claims about *why*
> something works. Where the manual and the package disagree, the package
> is correct; where the manual and the papers disagree, the papers are.

**Status: complete draft.** Every code block is executed by
`run_blocks.py`, which runs the fenced blocks of a chapter in one
namespace, in order:

```bash
python docs/manual/run_blocks.py docs/manual/*.md
```

A full pass takes about 25 minutes on a desktop GPU, most of it in five
chapters that train real models: 5 (a deep network), 13 and 15 (a
cross-validation each), 16 (a two-variable network) and 17 (two 3D
implicit models). Everything else runs in seconds.

Everything runs on the bundled sample data except chapter 17, which
fetches about 20 kB of drillhole logs on first run and caches them in
`docs/manual/data/`; it is skipped, not failed, when there is no
connection and nothing cached.

## Part I — From kriging to the variational GP

1. [Why another geostatistics](01-why-another-geostatistics.md)
2. [The GP is kriging](02-the-gp-is-kriging.md)
3. [Inducing points and the ELBO](03-inducing-points-and-the-elbo.md)
4. [Warpings, likelihoods, and where the Gaussian sits](04-warpings-likelihoods-and-the-gaussian.md)
5. [Latent networks](05-latent-networks.md)
6. [Categories and boundaries](06-categories-and-boundaries.md)
7. [Simulation](07-simulation.md)
8. [Change of support](08-change-of-support.md)

## Part II — The workflow

9. [From database to data](09-from-database-to-data.md)
10. [Containers and addressing](10-containers-and-addressing.md)
11. [Building and training](11-building-and-training.md)
12. [Prediction, blocks and surfaces](12-prediction-blocks-and-surfaces.md)
13. [Validation](13-validation.md)
14. [Reporting](14-reporting.md)

## Part III — Case studies

15. [Walker Lake](15-case-study-walker-lake.md)
16. [Jura](16-case-study-jura.md)
17. [A folded quartz vein in 3D](17-case-study-quartz-vein.md)
