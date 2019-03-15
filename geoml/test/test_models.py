import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import plotnine as p9
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as ptls
#from tensorboard import main as tb

import geoml

#%% base GP - walker lake
walker = pd.read_table("C:\\Dropbox\\Python\\Pacotes\\geoml\\geoml\\sample_data\\walker.dat")
walker_ex = pd.read_table("C:\\Dropbox\\Python\\Pacotes\\geoml\\geoml\\sample_data\\walker_ex.dat", sep=",")

point = geoml.data.Points2D(walker[["X", "Y"]], walker.drop(["X", "Y"], axis=1))
point_ex = geoml.data.Points2D(walker_ex[["X", "Y"]], walker_ex.drop(["X", "Y"], axis = 1))

gp = geoml.models.GP(
    sp_data = point, 
    variable = "V", 
    kernels = [
         geoml.kernels.SphericalKernel(geoml.transform.Isotropic(5)),
         geoml.kernels.SphericalKernel(geoml.transform.Anisotropy2D(100, 0.5, 0)),
         geoml.kernels.ConstantKernel(),
         geoml.kernels.LinearKernel()],
    warping = [geoml.warping.Scaling(positive = True), 
               geoml.warping.Softplus(), 
               geoml.warping.Spline(5)])
print(gp.log_lik())
gp.train(maxiter = 2500, seed = 1234)

# Warping
plt.figure()
plt.plot(gp.cov_model.warp_forward(np.sort(gp.y)), np.sort(gp.y), "-k")

# training log
plt.figure()
plt.plot(np.arange(gp.training_log["evolution"].shape[1]),
         gp.training_log["evolution"].max(0), "g-")
plt.plot(np.arange(gp.training_log["evolution"].shape[1]),
         gp.training_log["evolution"].mean(0), "r-")
plt.fill_between(np.arange(gp.training_log["evolution"].shape[1]),
                 gp.training_log["evolution"].max(0),
                 gp.training_log["evolution"].min(0),
                 facecolor='blue', alpha=0.5)

# prediction
gp.predict(point_ex, name="Vpred")
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(np.flip(np.reshape(point_ex.data["Vpred_p0.025"].values, (300, 260)), 0))
plt.colorbar()
plt.title("Vpred_p0.025")
plt.subplot(1, 3, 2)
plt.imshow(np.flip(np.reshape(point_ex.data["Vpred_p0.5"].values, (300, 260)), 0))
plt.colorbar()
plt.title("Vpred_p0.5")
plt.subplot(1, 3, 3)
plt.imshow(np.flip(np.reshape(point_ex.data["Vpred_p0.975"].values, (300, 260)), 0))
plt.colorbar()
plt.title("Vpred_p0.975")

# TensorFlow graph
gp.graph.get_operations()

path = "C:\\Dropbox\\Python\\Lab\\geoML\\viz\\gp"
writer = tf.summary.FileWriter(path, graph = gp.graph)

#tf.flags.FLAGS.logdir = path
#tb.main()

#%% GP with gradients - 2D example data

ex_point = pd.DataFrame(
        {"X" : np.array([25, 40, 60, 85,
                        5, 10, 45, 50, 55, 75, 90,
                        15, 20, 30, 50, 65, 75, 90, 25, 50, 65, 75]),
         "Y" : np.array([25, 60, 50, 15,
                        50, 80, 10, 30, 10, 75, 90,
                        15, 35, 65, 85, 65, 50, 20, 10, 50, 20, 10]),
         "val" : np.concatenate([np.repeat(1, 4), 
                           np.repeat(-1, 7),
                           np.repeat(0, 11)]),
         "val_str" : np.concatenate([np.repeat("inside", 4), 
                           np.repeat("outside", 7),
                           np.repeat("border", 11)])})
ex_point["label_1"] = pd.Categorical(np.concatenate(
        [np.repeat("a", 4), 
         np.repeat("b", 7),
         np.repeat("a", 11)]))
ex_point["label_2"] = pd.Categorical(np.concatenate(
        [np.repeat("a", 4), 
         np.repeat("b", 7),
         np.repeat("b", 11)]))
ex_point["interpolate"] = ex_point["val"] == 0

ex_dir = pd.DataFrame(
        {"X" : np.array([40, 50, 70, 90, 30, 20, 10]),
         "Y" : np.array([40, 85, 70, 30, 50, 60, 10]),
         "strike" : np.array([30, 90, 325, 325, 30, 30, 30])})
ex_dir["dX"] = np.sin(ex_dir["strike"]/180*np.pi)
ex_dir["dY"] = np.cos(ex_dir["strike"]/180*np.pi)
#ex_dir["a"] = np.ones_like(ex_dir["dY"].values)

point = geoml.data.Points2D(ex_point[["X", "Y"]], ex_point.drop(["X", "Y"], axis = 1))
dirs = geoml.data.Directions2D(ex_dir[["X", "Y"]], ex_dir[["dX", "dY"]])

gp = geoml.models.GPClassif(
        point, "label_1", "label_2", dir_data = dirs,
        kernels = [geoml.kernels.GaussianKernel(geoml.transform.Isotropic(60))])
ex_grid = geoml.data.Grid2D([0, 0], [101, 101], [1, 1])
gp.predict(ex_grid, name = "label")


###########
point, dirs = geoml.data.Examples.example_fold()
rng = 60

only_points = geoml.models.GPClassif(
                    sp_data=point, var_1="label_1", var_2="label_2",
                    kernels = [geoml.kernels.CubicKernel(geoml.transform.Isotropic(rng))])
points_and_directions = geoml.models.GPClassif(
                            sp_data=point, var_1="label_1", var_2="label_2", dir_data = dirs,
                            kernels = [geoml.kernels.CubicKernel(geoml.transform.Isotropic(rng))])

only_points.GPs[0].cov_model.variance.set_value(np.array([0.9,0.1]))
only_points.GPs[1].cov_model.variance.set_value(np.array([0.9,0.1]))
#only_points.GPs[0].cov_model.fix_kernel_parameter(0, "range")
#only_points.GPs[1].cov_model.fix_kernel_parameter(0, "range")
#only_points.train(seed=1234)

# coordinates to receive the predictions
grid = geoml.data.Grid2D(start=[0, 0], n=[101, 101], step=[1, 1])
only_points.predict(newdata=grid, name="model_1", verbose=False, n_samples=10000)
points_and_directions.predict(newdata=grid, name="model_2", verbose=False,
                              n_samples=10000)

ind_1 = grid.as_image("model_1_rock_a_ind")
ind_2 = grid.as_image("model_2_rock_a_ind")
df_point = point.as_data_frame()
df_dir = dirs.as_data_frame()

fig = plt.figure(figsize=(16,16))

ax1 = fig.add_subplot(1, 2, 1)
ax1.contour(grid.grid[0], grid.grid[1], ind_1, levels=15)#, colors="gray", linestyles="solid")
ax1.contour(grid.grid[0], grid.grid[1], ind_1, levels=np.array([0]), colors="black")
#im=ax1.imshow(ind_1, origin="lower")
#plt.colorbar(im, ax=ax1)
ax1.plot(df_point["X"].values[df_point["rock"] == "rock_a"], df_point["Y"].values[df_point["rock"] == "rock_a"], "bo")
ax1.plot(df_point["X"].values[df_point["rock"] == "rock_b"], df_point["Y"].values[df_point["rock"] == "rock_b"], "ro")
ax1.plot(df_point["X"].values[df_point["rock"] == "boundary"], df_point["Y"].values[df_point["rock"] == "boundary"], "ko")
ax1.set_aspect(1.0)

ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(grid.grid[0], grid.grid[1], ind_2, levels=15)#, colors="gray", linestyles="solid")
ax2.contour(grid.grid[0], grid.grid[1], ind_2, levels=np.array([0]), colors="black")
ax2.plot(df_point["X"].values[df_point["rock"] == "rock_a"], df_point["Y"].values[df_point["rock"] == "rock_a"], "bo")
ax2.plot(df_point["X"].values[df_point["rock"] == "rock_b"], df_point["Y"].values[df_point["rock"] == "rock_b"], "ro")
ax2.plot(df_point["X"].values[df_point["rock"] == "boundary"], df_point["Y"].values[df_point["rock"] == "boundary"], "ko")
ax2.set_aspect(1.0)

plt.show()



#%% Classification
ara_lito = pd.read_excel("C:\\Dropbox\\Python\\Pacotes\\geoml\\geoml\\sample_data\\Araranguá.xlsx", 
                    sheet_name="Lito")
ara_collar = pd.read_excel("C:\\Dropbox\\Python\\Pacotes\\geoml\\geoml\\sample_data\\Araranguá.xlsx", 
                    sheet_name="Collar")

ara_dh = geoml.data.DrillholeData(ara_collar, ara_lito, holeid="Hole ID", fr="From", to="To")
ara_point = ara_dh.as_classification_input("Formation", [0.05, 0.25, 0.5, 0.75, 0.95])

gp = geoml.models.GPClassif(
        ara_point, "Formation_up", "Formation_down",
        [geoml.kernels.CubicKernel(
            geoml.transform.Anisotropy3D(500, minrange_fct=0.1))])
for model in gp.GPs:
    model.cov_model.fix_kernel_parameter(0, "midrange_fct")
    model.cov_model.set_kernel_parameter_limits(0, "minrange_fct", 0.01, 0.2)
    model.cov_model.set_kernel_parameter_limits(0, "maxrange", 5000, 30000)
    model.cov_model.fix_kernel_parameter(0, "azimuth")
    model.cov_model.fix_kernel_parameter(0, "dip")
    model.cov_model.fix_kernel_parameter(0, "rake")
print(gp.log_lik())
gp.train(seed=1234)

ara_grid = geoml.data.GridData(647000, 71, 200,
                               6791000, 41, 200,
                               -550, 66, 10)
#ara_grid = geoml.data.GridData(647000, 141, 100,
#                               6791000, 81, 100,
#                               -550, 136, 5)
gp.predict(ara_grid, "Formation")

form_color = {"Recente": "rgb(18,135,38)",
              "Rio do Rasto": "rgb(39,96,165)",
              "Estrada Nova": "rgb(102,71,15)",
              "Irati": "rgb(130,40,25)",
              "Palermo": "rgb(105,16,117)",
              "Rio Bonito": "rgb(201,194,96)"}

holes = ara_dh.draw_categorical("Formation", form_color, line={"width": 10})
cont = ara_grid.draw_contour("Formation_Recente_ind", 0,
                              color = form_color["Recente"]) + \
        ara_grid.draw_contour("Formation_Rio do Rasto_ind", 0,
                              color = form_color["Rio do Rasto"]) + \
        ara_grid.draw_contour("Formation_Estrada Nova_ind", 0,
                              color = form_color["Estrada Nova"]) + \
        ara_grid.draw_contour("Formation_Irati_ind", 0,
                              color = form_color["Irati"]) + \
        ara_grid.draw_contour("Formation_Palermo_ind", 0,
                              color = form_color["Palermo"]) + \
        ara_grid.draw_contour("Formation_Rio Bonito_ind", 0,
                              color = form_color["Rio Bonito"])
py.plot(go.Figure(holes+[cont[i] for i in [1,2,4,5]], ara_point.aspect_ratio(10)))

pts = ara_point.draw_categorical("Formation_up", form_color)
py.plot(go.Figure(pts+[cont[i] for i in [1,2,4,5]], ara_point.aspect_ratio(10)))

mask = ara_grid.data["Formation_entropy"] > 0.92
ara_grid.data["Formation_filtered"] = ara_grid.data["Formation"]
ara_grid.data.loc[mask, "Formation_filtered"] = "None"
form_color2 = form_color.copy()
form_color2.update({"None": "rgb(125,125,125)"})
data = ara_grid.draw_section_numeric("Formation_entropy",
                                     axis = 0, visible = False)
#data = ara_grid.draw_section_categorical("Formation_filtered", colors = form_color2,
#                                     axis = 0, visible = False)
data[0].update({"visible": True})
    
steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',  # 'relayout'
        args = ['visible', [False] * len(data)],
        label = str(ara_grid.grid[0][i])
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)
    
sliders = [dict(
    active = 0,
    currentvalue = {"prefix": "X: "},
    pad = {"t": 50},
    steps = steps
)]
    
layout = geoml.data._update(dict(sliders=sliders), ara_grid.aspect_ratio(10))
py.plot(go.Figure(data, layout))
                    

