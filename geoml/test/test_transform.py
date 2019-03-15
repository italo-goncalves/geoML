import numpy as np
import tensorflow as tf
import pandas as pd
import plotnine as p9

import geoml

angle = np.array(np.linspace(0, 2*np.pi, 100), ndmin=2).transpose()
coords = np.append(np.cos(angle), np.sin(angle), axis=1)

tr_identity = geoml.transform.Identity()
tr_isotropic = geoml.transform.Isotropic(5)
tr_anis_2D = geoml.transform.Anisotropy2D(3, 1 / 3, 60)
tr_proj = geoml.transform.Projection2DTo1D(30, 1)

g = tf.Graph()
with g.as_default():
    tr_isotropic.params["range"].init_tf_placeholder()

    tr_anis_2D.params["maxrange"].init_tf_placeholder()
    tr_anis_2D.params["minrange_fct"].init_tf_placeholder()
    tr_anis_2D.params["azimuth"].init_tf_placeholder()

    tr_proj.params["azimuth"].init_tf_placeholder()
    tr_proj.params["range"].init_tf_placeholder()

    test_identity = tr_identity.forward(coords)
    test_isotropic = tr_isotropic.forward(coords)
    test_anis_2D = tr_anis_2D.forward(coords)
    test_proj = tr_proj.backward(coords)
    
    init = tf.global_variables_initializer()

feed = {}
feed.update(tr_isotropic.params["range"].tf_feed_entry)
feed.update(tr_anis_2D.params["maxrange"].tf_feed_entry)
feed.update(tr_anis_2D.params["minrange_fct"].tf_feed_entry)
feed.update(tr_anis_2D.params["azimuth"].tf_feed_entry)
feed.update(tr_proj.params["azimuth"].tf_feed_entry)
feed.update(tr_proj.params["range"].tf_feed_entry)
with tf.Session(graph=g) as session:
    session.run(init, feed_dict=feed)
    out_identity = test_identity.eval(session=session, feed_dict=feed)
    out_isotropic = test_isotropic.eval(session=session, feed_dict=feed)
    out_anis_2D = test_anis_2D.eval(session=session, feed_dict=feed)
    out_proj = test_proj.eval(session=session, feed_dict=feed)

# visual testing
df = pd.DataFrame({"x": test_identity[:, 0], "y": test_identity[:, 1]})
fig_id = p9.ggplot(df) +\
         p9.geom_point(p9.aes("x", "y")) +\
         p9.coord_fixed()
        
df = pd.DataFrame({"x": test_isotropic[:, 0], "y": test_isotropic[:, 1]})
fig_iso = p9.ggplot(df) +\
          p9.geom_point(p9.aes("x", "y")) +\
          p9.coord_fixed()
        
df = pd.DataFrame({"x": test_anis_2D[:, 0], "y": test_anis_2D[:, 1]})
fig_anis2D = p9.ggplot(df) +\
             p9.geom_point(p9.aes("x", "y")) +\
             p9.coord_fixed()
