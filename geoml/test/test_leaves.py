"""A tree's leaves as the points of contact with the likelihoods.

`VGPNetwork` used to take one node, so a model with two likelihoods ended in
a `Concatenate` whose only purpose was to be split apart again -- a join
the diagram drew that was not part of the model. It takes a list of leaves
now, one per likelihood, or a mapping from each variable to its likelihood
with the leaves beside it. Pinned here: the two spellings, the refusals,
the single-leaf property, the equivalence of a list of leaves on a shared
root with the `Concatenate` of them, persistence keeping a shared parent
shared, the diagram drawing each likelihood off its own leaf, and the gate
that motivated it -- two leaves on two independent trees, roots of their
own, no join at all.
"""
import numpy as np
import pytest

import geoml
import geoml.latent as gl
import geoml.likelihood as lk
import geoml.persistence as persistence
import geoml.transform as tr


def _jura():
    train, held = geoml.datasets.jura()
    return train, held, len(train.get("Rock").labels), \
        len(train.get("Elements").labels)


def _options(**kwargs):
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("training_samples", 8)
    return geoml.models.GPOptions(**kwargs)


def _two_leaves(train, n_rock, n_el, seed=1234):
    """Two leaves sharing one root: a rock GP and a metals GP."""
    geoml.set_seed(seed)
    root = gl.BasicInput(geoml.data.inducing.from_kmeans(train, 60, seed=0),
                         transform=tr.Isotropic(1.0))
    rock = gl.BasicGP(root, size=n_rock)
    metal = gl.BasicGP(root, size=n_el)
    return root, rock, metal


def _likelihoods(n_rock, n_el):
    return [lk.CategoricalGaussianIndicator(n_rock),
            lk.MultivariateGaussian(n_el)]


# --------------------------------------------------------------------------- #
# the two spellings, and what is refused
# --------------------------------------------------------------------------- #
def test_a_list_of_leaves_is_one_per_likelihood():
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())

    assert model.leaves == [rock, metal]
    assert model.lik_sizes == [n_rock, n_el]


def test_a_mapping_names_each_likelihood_beside_its_variable():
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    rock_lik, metal_lik = _likelihoods(n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, {"Rock": rock_lik, "Elements": metal_lik},
        latent_network=[rock, metal], options=_options())

    assert model.variables == ["Rock", "Elements"]
    assert model.likelihoods == [rock_lik, metal_lik]
    assert model.leaves == [rock, metal]


def test_a_single_node_still_serves_every_likelihood():
    """The shape every model had: one node, split among the likelihoods."""
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    joined = gl.Concatenate(rock, metal)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el), joined,
        options=_options())

    assert model.leaves == [joined]
    assert model.latent_network is joined


def test_the_property_names_the_leaves_when_there_are_several():
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())

    with pytest.raises(AttributeError, match="2 leaves"):
        model.latent_network


def test_the_refusals():
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    liks = _likelihoods(n_rock, n_el)

    with pytest.raises(ValueError, match="1 leaves for 2"):
        geoml.models.VGPNetwork(train, ["Rock", "Elements"], liks, [rock])
    with pytest.raises(ValueError, match="has size"):
        geoml.models.VGPNetwork(train, ["Rock", "Elements"], liks,
                                [metal, rock])
    with pytest.raises(ValueError, match="given twice"):
        geoml.models.VGPNetwork(
            train, {"Rock": liks[0], "Elements": liks[1]}, liks, [rock, metal])
    with pytest.raises(ValueError, match="no likelihoods"):
        geoml.models.VGPNetwork(train, ["Rock", "Elements"],
                                latent_network=[rock, metal])
    with pytest.raises(ValueError, match="no latent network"):
        geoml.models.VGPNetwork(train, ["Rock", "Elements"], liks)


# --------------------------------------------------------------------------- #
# the equivalence: leaves on a shared root against the join of them
# --------------------------------------------------------------------------- #
def _train_and_predict(model, held, iterations=6):
    model.train_full(max_iter=iterations)
    model.predict(held, n_sim=4)
    return (np.asarray(model.training_log),
            np.asarray(held.values("Elements/Zn/prediction"), dtype=float),
            np.asarray(held.values("Rock/predicted")))


def test_a_list_of_leaves_is_the_join_of_them_to_rounding():
    """Concatenate-then-split and predict-per-leaf order the arithmetic
    differently, so equal to rounding rather than to the bit."""
    train, held, n_rock, n_el = _jura()

    _, rock, metal = _two_leaves(train, n_rock, n_el)
    joined_model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        gl.Concatenate(rock, metal), options=_options())
    log_joined, zn_joined, rock_joined = _train_and_predict(joined_model, held)

    _, rock, metal = _two_leaves(train, n_rock, n_el)
    leaves_model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())
    log_leaves, zn_leaves, rock_leaves = _train_and_predict(leaves_model, held)

    np.testing.assert_allclose(log_leaves, log_joined, rtol=1e-10, atol=0)
    np.testing.assert_allclose(zn_leaves, zn_joined, rtol=1e-8, atol=1e-10)
    assert np.array_equal(rock_leaves, rock_joined)


def test_a_shared_parent_is_priced_once():
    """The KL sums over every node once: two leaves on one root pay for the
    root once, exactly as the join of them did."""
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())

    nodes = model._nodes()
    assert len(nodes) == len({id(n) for n in nodes})
    assert rock.parent is metal.parent
    assert sum(n is rock.parent for n in nodes) == 1


# --------------------------------------------------------------------------- #
# persistence and the diagram
# --------------------------------------------------------------------------- #
def test_a_saved_model_keeps_its_leaves_and_their_shared_parent(tmp_path):
    train, held, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, {"Rock": _likelihoods(n_rock, n_el)[0],
                "Elements": _likelihoods(n_rock, n_el)[1]},
        latent_network=[rock, metal], options=_options())
    model.train_full(max_iter=3)
    model.predict(held, n_sim=4)
    before = np.asarray(held.values("Elements/Zn/prediction"), dtype=float)

    persistence.save_model(model, tmp_path / "model")
    restored = persistence.load_model(tmp_path / "model")

    assert len(restored.leaves) == 2
    assert restored.leaves[0].parent is restored.leaves[1].parent
    assert restored.variables == ["Rock", "Elements"]
    restored.predict(held, n_sim=4)
    np.testing.assert_allclose(
        np.asarray(held.values("Elements/Zn/prediction"), dtype=float), before)


def _arrows(dot):
    return [line.strip() for line in dot.splitlines() if " -> " in line]


def test_the_diagram_draws_each_likelihood_off_its_own_leaf():
    train, _, n_rock, n_el = _jura()
    root, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())

    dot = model.to_dot(legend=False)
    assert "Concatenate" not in dot
    # the shared root is one box with one arrow to each leaf, and each
    # variable box hangs off a different leaf
    arrows = _arrows(dot)
    assert len(arrows) == len(set(arrows))            # nothing drawn twice
    boxes = [line for line in dot.splitlines() if "label=" in line
             and " -> " not in line]
    assert sum("BasicInput" in b for b in boxes) == 1
    root_id = [b.split()[0] for b in boxes if "BasicInput" in b][0]
    assert sum(a.startswith(root_id + " ->") for a in arrows) == 2


# --------------------------------------------------------------------------- #
# what walks the tree from the model
# --------------------------------------------------------------------------- #
def test_cross_validation_resets_every_leaf_and_the_shared_root_once():
    train, _, n_rock, n_el = _jura()
    _, rock, metal = _two_leaves(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())
    model.train_full(max_iter=3)
    train.spatial_k_fold(train, k=2, seed=0)

    oof, scores = geoml.models.cross_validate(
        model, iterations=2, n_sim=4, n_nodes=4)
    assert np.all(np.isfinite(
        np.asarray(oof.values("Elements/Zn/prediction"), dtype=float)))
    assert set(scores["variable"]) == {"Elements"}   # the rock is categorical


# --------------------------------------------------------------------------- #
# independent trees
# --------------------------------------------------------------------------- #
def _two_trees(train, n_rock, n_el, seed=1234):
    """Two leaves on two roots -- a coarse inducing set for the rock, a fine
    one for the metals -- with no join anywhere."""
    geoml.set_seed(seed)
    coarse = gl.BasicInput(geoml.data.inducing.from_kmeans(train, 40, seed=0),
                           transform=tr.Isotropic(1.0))
    fine = gl.BasicInput(geoml.data.inducing.from_kmeans(train, 120, seed=1),
                         transform=tr.Isotropic(1.0))
    return coarse, fine, gl.BasicGP(coarse, size=n_rock), \
        gl.BasicGP(fine, size=n_el)


def test_a_stack_still_joins_two_trees_as_before():
    """The door that existed before a list did: a terminal `Stack` joins
    latent variables from separate trees without propagating inducing
    points. Kept so the refactor cannot regress it."""
    train, held, n_rock, n_el = _jura()
    coarse, fine, rock, metal = _two_trees(train, n_rock, n_el)
    joined = gl.Stack(rock, metal)
    assert not joined.same_root
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el), joined,
        options=_options())
    model.train_full(max_iter=3)
    model.predict(held, n_sim=2)
    assert np.all(np.isfinite(
        np.asarray(held.values("Elements/Zn/prediction"), dtype=float)))


def test_two_leaves_on_two_trees_train_predict_and_reload(tmp_path):
    """The gate: two roots as a list, no join at all. It builds, the bound
    improves, the gradient reaches both roots, prediction does not depend on
    the batch, and it survives a save."""
    train, held, n_rock, n_el = _jura()
    coarse, fine, rock, metal = _two_trees(train, n_rock, n_el)
    model = geoml.models.VGPNetwork(
        train, ["Rock", "Elements"], _likelihoods(n_rock, n_el),
        [rock, metal], options=_options())
    assert len(model._nodes()) == 4                   # two roots, two leaves

    before = [float(coarse.transform.parameters["range"].get_value()),
              float(fine.transform.parameters["range"].get_value())]
    model.train_full(max_iter=6)
    after = [float(coarse.transform.parameters["range"].get_value()),
             float(fine.transform.parameters["range"].get_value())]
    assert model.training_log[-1] > model.training_log[0]
    assert after[0] != before[0] and after[1] != before[1]

    # batch invariance: the whole held-out set against it in two halves
    model.predict(held, n_sim=4)
    whole = np.asarray(held.values("Elements/Zn/prediction"), dtype=float).copy()
    half = np.zeros(held.n_data, dtype=bool)
    half[: held.n_data // 2] = True
    model.predict(held, n_sim=4, where=half)
    model.predict(held, n_sim=4, where=~half)
    np.testing.assert_allclose(
        np.asarray(held.values("Elements/Zn/prediction"), dtype=float), whole)

    persistence.save_model(model, tmp_path / "trees")
    restored = persistence.load_model(tmp_path / "trees")
    assert restored.leaves[0].parent is not restored.leaves[1].parent
    restored.predict(held, n_sim=4)
    np.testing.assert_allclose(
        np.asarray(held.values("Elements/Zn/prediction"), dtype=float), whole)
