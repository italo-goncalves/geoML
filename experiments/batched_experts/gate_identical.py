"""Step 3's gate: the expert subset changes nothing when it holds them all.

Run under two package trees (PYTHONPATH) and compare the printed hex:
the unchanged package, and this branch with and without
`expert_subset(all)`. Training is `train_full` and `train_svi`; the
prediction is `predict` at the test points; deep (a GP on a GP) and
shallow models, both propagation rules.
"""
import sys

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

subset = sys.argv[1] == "subset" if len(sys.argv) > 1 else False
J = 4


def digest(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    import hashlib
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


for deep, rule in ((False, "consensus"), (True, "consensus"),
                   (True, "independent")):
    data, test, yt, truth = common.dataset(J)
    m = common.model(data, J, deep=deep, propagation=rule)

    def run():
        m.train_full(20)
        m.train_svi(1)
        m.predict(test, n_sim=10)

    if subset:
        with geoml.latent.expert_subset(range(J)):
            run()
    else:
        run()
    print("deep=%s rule=%s log %s pred %s sims %s" % (
        deep, rule, digest(m.training_log),
        digest(test.values("v/prediction")),
        digest(test.variables["v"].get_simulations())), flush=True)
