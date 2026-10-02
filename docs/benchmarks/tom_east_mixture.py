"""A mixture of likelihoods on Tom East's three metals, 2026-10-01.

Usage: python docs/benchmarks/tom_east_mixture.py <path to Tom East.csv>

The file is not in the repository: it is part of the Macpass data the
package does not bundle. Its intervals carry Ag (g/t), Pb and Zn (%), and
`Code_Simple`, the logged rock type -- "Tom East" ore or "Waste".

The 798 intervals assayed for all three metals are modelled as one vector
variable, and the models compared by cross-validation, folds by hole
(`spatial_k_fold` grouped on `HOLEID`, the prediction target the assayed
intervals themselves -- against every interval logged, most of them in
holes with no assay, the folds mimicked distances so long that the largest
held out over half the data and the rock type itself was predicted out of
fold at little better than chance):

- **single**: one Gaussian likelihood through the vector chain the
  package recommends, `BoxCox -> RobustPCA -> ZScore -> SinhArcsinh ->
  ZScore`, on a leaf of three latent columns;
- **plain**: one Gaussian through `BoxCox -> ZScore`, the chain each
  mixture component has, so that the mixture is compared with the same
  warping unmixed as well as with the recommended one;
- **mixture**: `LikelihoodMixture` of two Gaussians, each through its own
  `BoxCox -> ZScore`, on a leaf of six, the shares fixed;
- **latent**: the same, the shares read from a GP of their own, two
  columns, learned from the metals alone, sharpened by a trained
  amplitude and moved by a trained bias per share;
- **latent_rock**: the shares from a GP node of their own, which also
  feeds a categorical likelihood on `Code_Simple` (its labels ordered
  Waste, Tom East, as the populations start: the lower group first; a
  trained bias per category), and is concatenated with the populations' GP
  into the mixture's leaf. The rock type is trained on every logged
  interval, 4585, the metals missing where unassayed: on the assayed ones
  alone it is 52.5% Tom East, against 11.0% logged -- the assays were taken
  where the ore is -- and its bias had no background to learn.

The expected shares are also scored out of fold against the logged rock
type: a share field predicts at locations it never saw, so its agreement
is honest only there. The bar is the rock type's own out-of-fold
prediction in the `latent_rock` arm, not chance: the logged rock type is
barely predictable between holes here, and a share field cannot know more
about it than a likelihood trained on it.

Every arm reads one root: 500 k-means centroids of the assayed intervals
divided into five experts overlapping by 10%, through an anisotropy of
100 m along azimuth 345, dip 15, rake 70 (middle and minor ranges 0.75 and
0.5 of it). Cross-validation drops only the held-out data; the inducing
points stay. The 23 holes with no assay join the fold of the nearest
assayed hole, so the folds of the assayed intervals are the same in every
arm, and every agreement is read on the assayed intervals.

Reported: the pooled out-of-fold rmse, CRPS and interval goodness per
metal, and how often each interval's largest responsibility agrees with
its logged rock type (in sample, under the better naming of the two).
"""
import sys
import time

import numpy as np
import pandas as pd

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.transform as tr
import geoml.warping as wp

METALS = ["Ag", "Pb", "Zn"]
# the order of the smallest assays, so a value at the detection limit is
# not sent to minus infinity by the logarithm inside Box-Cox
SHIFT = 0.005


def load(path):
    frame = pd.read_csv(path)
    every = geoml.data.PointData(frame, ["X", "Y", "Z"])
    assayed = frame.dropna(subset=METALS).reset_index(drop=True)
    data = geoml.data.PointData(assayed, ["X", "Y", "Z"])
    data.add_vector_variable("metals", METALS, assayed[METALS].values)
    data.add_metadata("HOLEID", assayed["HOLEID"].values)
    data.add_metadata("ore", (assayed["Code_Simple"] == "Tom East").values)
    data.add_categorical_variable("rock", labels=["Waste", "Tom East"],
                                  measurements=assayed["Code_Simple"].values)
    return data, every


def load_logged(path, assayed):
    """Every logged interval, the metals missing where not all three were
    assayed, and the folds of `assayed` carried over by hole -- a hole with
    no assay taking the fold of the nearest assayed one, by the holes'
    mean positions."""
    frame = pd.read_csv(path)
    metals = frame[METALS].values.copy()
    measured = ~np.isnan(metals).any(axis=1)
    metals[~measured] = np.nan
    data = geoml.data.PointData(frame, ["X", "Y", "Z"])
    data.add_vector_variable("metals", METALS, metals)
    data.add_categorical_variable("rock", labels=["Waste", "Tom East"],
                                  measurements=frame["Code_Simple"].values)
    data.add_metadata("HOLEID", frame["HOLEID"].values)
    data.add_metadata("ore", (frame["Code_Simple"] == "Tom East").values)
    data.add_metadata("assayed", measured)

    folds = pd.Series(np.asarray(assayed.get_metadata("fold")),
                      index=np.asarray(assayed.get_metadata("HOLEID")))
    folds = folds.groupby(level=0).first()
    centres = frame.groupby("HOLEID")[["X", "Y", "Z"]].mean()
    known = centres.loc[folds.index]
    fold_of = {}
    for hole, centre in centres.iterrows():
        if hole in folds.index:
            fold_of[hole] = folds[hole]
        else:
            nearest = np.argmin(np.sum((known.values - centre.values) ** 2,
                                       axis=1))
            fold_of[hole] = folds.iloc[nearest]
    data.add_metadata("fold", frame["HOLEID"].map(fold_of).values)
    return data


def likelihood(arm):
    if arm == "single":
        return lk.Gaussian(wp.ChainedWarping(
            wp.BoxCox(3, shift=SHIFT), wp.RobustPCA(3), wp.ZScore(3),
            wp.SinhArcsinh(3), wp.ZScore(3)))
    if arm == "plain":
        return lk.Gaussian(wp.ChainedWarping(wp.BoxCox(3, shift=SHIFT),
                                             wp.ZScore(3)))
    return lk.LikelihoodMixture([
        lk.Gaussian(wp.ChainedWarping(wp.BoxCox(3, shift=SHIFT),
                                      wp.ZScore(3)))
        for _ in range(2)],
        shares="fixed" if arm == "mixture" else "latent")


def model(arm, data, assayed=None):
    """`arm` trained on `data`, its inducing points from `assayed` -- the
    assayed intervals -- where `data` holds more than those."""
    geoml.set_seed(1234)
    ip = geoml.data.inducing.experts(
        geoml.data.inducing.from_kmeans(
            data if assayed is None else assayed, 500, seed=0), 5,
        overlap=0.1, seed=0)
    root = latent.BasicInput(
        ip, transform=tr.Anisotropy3D(100, 0.75, 0.5, 345, 15, 70))
    options = geoml.models.GPOptions(verbose=False, training_samples=20)
    if arm == "latent_rock":
        populations = latent.BasicGP(root, size=6)
        shares = latent.BasicGP(root, size=2)
        m = geoml.models.VGPNetwork(
            data, {"metals": likelihood(arm),
                   "rock": lk.CategoricalGaussianIndicator(2, bias=True)},
            latent_network=[latent.Concatenate(populations, shares), shares],
            options=options)
    elif arm == "latent":
        m = geoml.models.VGPNetwork(
            data, "metals", likelihood(arm),
            latent.Concatenate(latent.BasicGP(root, size=6),
                               latent.BasicGP(root, size=2)),
            options=options)
    else:
        lik = likelihood(arm)
        m = geoml.models.VGPNetwork(
            data, "metals", lik, latent.BasicGP(root, size=lik.size),
            options=options)
    m.train_full(max_iter=800)
    return m


def agreement(m, data):
    responsibilities = m.responsibilities(data, store=False)["metals"]
    ore = np.asarray(data.get_metadata("ore"), dtype=bool)
    first = responsibilities[:, 0] > 0.5
    return max(np.mean(first == ore), np.mean(first != ore))


def _assayed(oof):
    """The rows of the assayed intervals, all of them where the container
    holds nothing else."""
    if "assayed" not in oof.metadata:
        return np.ones(oof.n_data, dtype=bool)
    return np.asarray(oof.get_metadata("assayed"), dtype=bool)


def oof_agreement(oof):
    """How often the out-of-fold expected share of the higher population
    agrees with the logged rock type, on the assayed intervals."""
    rows = _assayed(oof)
    shares = oof.variables["metals"].responsibilities
    second = shares[1].values.to_numpy()[rows] > 0.5
    ore = np.asarray(oof.get_metadata("ore"), dtype=bool)[rows]
    return np.mean(second == ore)


def rock_agreement(oof):
    """How often the rock type's own out-of-fold probability agrees with
    the logged rock type -- the bar the shares are held to -- and how often
    it agrees with the shares themselves; on the assayed intervals."""
    rows = _assayed(oof)
    rock = oof.variables["rock"].components["Tom East"] \
        .probability.values.to_numpy()[rows] > 0.5
    share = oof.variables["metals"].responsibilities[1] \
        .values.to_numpy()[rows] > 0.5
    ore = np.asarray(oof.get_metadata("ore"), dtype=bool)[rows]
    return np.mean(rock == ore), np.mean(rock == share)


def rock_scores(oof, rows):
    """Out of fold over `rows`: for the rock type's own call and for the
    expected shares, the share of intervals agreeing with the log, each
    category's recall, and their mean (balanced accuracy, which a model
    calling every interval waste cannot pass off as skill)."""
    ore = np.asarray(oof.get_metadata("ore"), dtype=bool)[rows]
    calls = {
        "rock": oof.variables["rock"].components["Tom East"]
        .probability.values.to_numpy()[rows] > 0.5,
        "shares": oof.variables["metals"].responsibilities[1]
        .values.to_numpy()[rows] > 0.5}
    out = {}
    for name, call in calls.items():
        waste, tom = np.mean(~call[~ore]), np.mean(call[ore])
        out[name] = "accuracy %.3f, recall Waste %.3f, Tom East %.3f, " \
            "balanced %.3f" % (np.mean(call == ore), waste, tom,
                               (waste + tom) / 2)
    return out


def main(path):
    data, _ = load(path)
    data.spatial_k_fold(data, k=5, groups="HOLEID")
    print("intervals %d, ore %d, fold sizes %s" % (
        data.n_data, int(np.sum(np.asarray(data.get_metadata("ore"),
                                            dtype=bool))),
        np.unique(np.asarray(data.get_metadata("fold")),
                  return_counts=True)[1]))
    arms = sys.argv[2:] or ("single", "plain", "mixture", "latent",
                            "latent_rock")
    for arm in arms:
        start = time.time()
        if arm == "latent_rock":
            logged = load_logged(path, data)
            m = model(arm, logged, assayed=data)
        else:
            m = model(arm, data)
        trained = time.time() - start
        oof, scores = geoml.models.cross_validate(m, n_sim=40)
        pooled = scores[(scores["fold"] == "all")
                        & (scores["variable"] == "metals")]
        line = "%-8s train %4.0f s, all %4.0f s" % (
            arm, trained, time.time() - start)
        if arm != "single" and arm != "plain":
            line += ", in-sample agreement %.3f, out of fold %.3f" % (
                agreement(m, data), oof_agreement(oof))
        if "amplitude" in m.likelihoods[0].parameters:
            line += ", amplitude %.2f" % float(
                m.likelihoods[0].parameters["amplitude"].get_value())
            bias = np.asarray(m.likelihoods[0].parameters["bias"]
                              .get_value())
            line += ", shares at zero %s" % np.round(
                np.exp(bias) / np.exp(bias).sum(), 3)
        if arm == "latent_rock":
            line += (", rock out of fold %.3f, rock and shares agree "
                     "%.3f" % rock_agreement(oof))
            line += ", rock bias %s" % np.round(np.asarray(
                m.likelihoods[1].parameters["bias"].get_value()), 2)
            for label, rows in (("assayed", _assayed(oof)),
                                ("all logged", np.ones(oof.n_data, bool))):
                for name, text in rock_scores(oof, rows).items():
                    line += "\n  %-10s %-6s %s" % (label, name, text)
        print(line)
        print(pooled[["component", "n", "rmse", "crps", "goodness"]]
              .to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
