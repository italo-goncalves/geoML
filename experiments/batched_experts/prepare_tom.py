"""The real case: Tom's 2 m composites, prepared as the Tom v6 notebook
prepares them, saved to `local_data/tom_composites.npz` (not in the
repository: Macpass data are not ours to publish)."""
import glob
import os

import numpy as np

import geoml

path = glob.glob("/mnt/c/Users/*talo/OneDrive/Micromine/Macpass")[0]
raw = geoml.datasets.macpass(path)
leg = raw.intervals["litho"].category_legend("Code")
leg["group"] = "Discard"
leg.loc[leg["label"].str.contains("TE"), "group"] = "Ore"
leg.loc[leg["label"].str.contains("TS"), "group"] = "Ore"
data = raw.group_categories(["litho", "Code"], groups=leg, new_column="Rock")
data = data.fill_unlogged(["litho", "Rock"], "Waste")
data = data.subset_region(min_val=[441400, 7003000],
                          max_val=[442600, 7004800])
data = data.rename("assay", {"Ag_ppm": "Ag", "Pb_pct": "Pb", "Zn_pct": "Zn"})
data.intervals["assay"].set_role("BD_tonnes_m3", "density")
comp = data.composite(length=2.0, domain=["litho", "Rock"])
points = comp.as_point_data(tables=["litho", "assay"], drop_missing=False)
print(points.tree())

zn = np.asarray(points.values("Zn/measurements"), dtype=float)
# the rock type is stored as integer codes against its labels; save the
# labels themselves, "" where nothing was logged
rock_var = points.variables["Rock"]
codes = np.asarray(rock_var.measurements_a.values).astype(int)
labels = np.asarray(list(rock_var.labels) + [""])
rock = labels[np.where(codes < 0, len(labels) - 1, codes)]
print("rock labels", list(rock_var.labels),
      {k: int((rock == k).sum()) for k in np.unique(rock)})
holes = np.asarray(points.get_metadata("HOLEID")).astype(str)
os.makedirs("local_data", exist_ok=True)
np.savez("local_data/tom_composites.npz",
         coords=np.asarray(points.coordinates), zn=zn, rock=rock.astype(str),
         holes=holes)
print("composites %d, with Zn %d, holes %d"
      % (len(zn), int(np.isfinite(zn).sum()), len(set(holes))))
