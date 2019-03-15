import pandas as pd

import geoml

walker = pd.read_table("C:\Dropbox\Python\Pacotes\geoml\geoml\sample_data\walker.dat")

point = geoml.data._PointData(walker[["X", "Y"]], walker.drop(["X", "Y"], axis = 1))
point

point = geoml.data._PointData(walker[["X", "Y"]])
point
