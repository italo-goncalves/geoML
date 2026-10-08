"""Does the unchanged package repeat itself on the gate's model? Prints
the bound after the first steps at full precision, so two runs show where
they part."""
import sys

import numpy as np

sys.path.insert(0, "experiments/batched_experts")
import common  # noqa: E402
import geoml  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "det":
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
data, test, yt, truth = common.dataset(4)
m = common.model(data, 4)
p0 = [np.asarray(v).ravel()[:3].tolist() for v in m.get_unfixed_variables()[:2]]
print("init", repr(p0))
m.train_full(20)
print("full", repr(float(m.training_log[-1])))
m.train_svi(1)
print("svi", [repr(float(v)) for v in m.training_log[20:]])
m.predict(test, n_sim=10)
print("pred", repr(float(np.sum(test.values("v/prediction")))),
      repr(float(np.sum(test.variables["v"].get_simulations()))))
