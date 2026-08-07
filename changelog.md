## version 0.5.6
* `geoml.inducing` builds the inducing points a network is given, which until
now had to be assembled by hand. `from_kmeans` puts them where the data is,
`from_grid` on a regular lattice, and `combine` merges sets — the usual
mixture of a regular backbone and the data's own locations is
`combine(from_grid(data, 50), from_kmeans(data, 200))`. The two `*_experts`
functions build a list, one set per expert, which is what `BasicInput` wants
when a model is divided into local experts. Both make neighbouring experts
overlap, which is what stops a prediction showing a seam where one gives way
to the next. `grid_experts` cuts space into regular blocks, each taking its own
block plus one node of margin all round: every expert holds the same number of
points and its neighbours are known in advance rather than measured — the
Moore neighbourhood, 8 in the plane and 26 in space. `experts` is the unordered
counterpart, for points that follow the data rather than a lattice, as a
drillhole survey does since it does not fill its bounding box. It clusters the
points into groups of about the same size, then lets each group take in
everything within `1 + overlap` times the Mahalanobis radius its own members
reach — a point absorbed that way keeps its original group as well, so the
experts come to share the points between them. The usual call divides a set
chosen beforehand, `experts(from_kmeans(data, 1500), 12)`. A cluster of samples
taken along one drillhole is nearly a line, whose covariance is singular and
whose Mahalanobis distance across the line would be unbounded, so the
eigenvalues are floored before the radius is measured
* A GP node no longer propagates its inducing points when nothing is built on
top of it. Every expert's set is predicted from every other, so this is the
one step in the network that is quadratic in the number of experts, and on the
last node of a network it was pure waste: nothing ever read the result. On a
single-layer model with 32 experts that alone took a prediction from 1.9 s to
0.18 s. Deeper networks keep paying it where it is genuinely needed
* The refresh that `predict` runs once per call is now traced rather than run
in eager Python. It is arithmetic over parameters that do not move during a
prediction, but run eagerly it pays Python overhead for each covariance block,
which with several experts was most of the call. The trace is kept on the
network so predicting again does not rebuild it, and it reads the parameters
live, so a model trained further still predicts from its new state
* A training step — the ELBO, its gradient and the update — is now traced as
one graph instead of being driven from eager Python. Run eagerly, Adam issues
its update variable by variable at about 1.5 ms each whatever the variable's
size, which is invisible on a small model and dominant on a large one: a GP
node holds three parameters per expert, so 40 experts spent more time in the
optimizer than in the model, 278 ms a step against 98 ms traced. The step is
built once per call to `train_full`/`train_svi` and reused. The cost is a
larger graph to build the first time, which a short run does not repay — the
mixed Jura case, five iterations, went from 30.3 s to 38.5 s — while a real
run recovers it within a few hundred iterations
* `train_svi` passes its directional data correctly. It was building each of
the four directional tensors with a trailing comma, so a one-element tuple
reached the model in place of the tensor
* Networks with more than one expert are now tested — building, training,
predicting in one batch and in many, propagating through a second layer, and
surviving a save and load

## version 0.5.5
* `get_contour(value, close="above")` closes a surface where it runs out of
the grid, so what comes back is a body with a volume rather than a sheet with
a hole in its side. It works by padding the cube with one cell of "well
outside" before contouring, which gives the surface somewhere to close inside
the padded volume — no capping geometry, and nothing to go wrong at a corner.
`"above"` keeps the region where the values exceed the contour, a grade shell;
`"below"` the region under it
* `DTM3D` is a terrain: a sheet that promises never to fold back over itself,
so that it stands at one height over each (x, y). That promise is what lets a
body be divided into what lies under it and what lies over it, and it is
checked where the object is made, by asking whether every triangle projects
onto the ground the same way round. Not what `mesh3d` returns — an ordinary
sheet stays a `Surface3D` until a terrain is asked for, this being a promise
to make rather than a fact to detect
* Meshes of different kinds can now be cut against each other, and the answer
keeps the caller's kind. `surface.intersection(solid)` is the part of the
sheet inside the body and `surface.difference(solid)` the part outside, both
sheets, cut where they cross the body rather than along their own triangles.
`solid.intersection(sheet)` is the part of the body below the sheet and
`solid.difference(sheet)` the part above, both bodies — worked by extruding
the sheet downwards into the ground beneath itself and letting the ordinary
body-to-body operations do the rest
* The edge cases raise rather than answer quietly, through four new errors,
all of them `ValueError`: `NotClosedError`, `InconsistentMeshError`,
`NotSingleValuedError` (a sheet that folds has no single "below" to keep) and
`MeshTypeError` (a sheet has no volume to add to a body; a sheet that does not
reach across a body would cut it at its own edge, and says so with both
extents in the message)
* Triangulated meshes are now three classes rather than one. `Mesh3D` is the
primitive — vertices, triangles, normals — and it measures itself as it is
built: its `area`, and whether it is `closed` and `consistent`, meaning its
triangles agree which way is out. `Surface3D` and `Solid3D` are siblings on
top of it, a sheet and a body, each checking its promise where the object is
made. A `Solid3D` therefore always has a `volume` worth reading, and
`assign_from_solid` can trust what it is handed instead of measuring it again
* `mesh3d(points, triangles, normals)` returns whichever of the three the
geometry calls for, and is what `get_contour` and `Mesh3D.from_dxf` now answer
with — so a contour that closes inside the grid comes back as a body, with its
volume, while one the grid cuts off comes back as a sheet
* A body wound inwards is turned round as it is built rather than refused:
nothing about it is ambiguous, only reversed. A closed mesh whose triangles
disagree with each other is a plain `Mesh3D`, being neither sheet nor body
* `Mesh3D.heal()` returns a repaired copy — vertices welded, holes up to a
size covered over, triangles made to agree and turned to face outward. That
last step is not a nicety: filling a hole leaves the new triangles wound
however they came, which would leave the mesh closed and still untestable.
What comes back is whichever class the repair earned, and an empty `Mesh3D` if
nothing survived
* `Mesh3D.split()` returns the connected pieces, each as an object of its own.
A boolean readily answers with a body in several pieces — an ore shell cut in
two — which is one legitimate mesh until you want them apart
* `Solid3D` does `union`, `intersection` and `difference`. VTK is asked first,
but it answers with nothing at all whenever the two surfaces have no face
crossing another — whether they stand apart or one contains the other, and
with no error either way — so an empty answer is not believed but worked out
from which body contains which. A difference where the other body lies wholly
inside is a body with a cavity, written as both surfaces with the inner one
turned inwards, so the volumes subtract and a location in the hollow tests as
outside
* A `Surface3D` can now be read from and written to a DXF file —
`Surface3D.from_dxf(filename)` and `surface.export_dxf(filename)`. Until now a
surface could only be *made*, by contouring a grid; a triangulation somebody
else built had no way in
* The export writes a single `MESH` entity, which holds the vertex list and the
triangles that index into it, so a surface comes back exactly as it went out.
The `POLYFACE` mesh that DXF triangulations are more often written as would
have been the obvious choice, but its vertex indices are 16-bit in most
readers — a ceiling near 32,000 vertices, which a contour off a real grid goes
through without trying
* The import is more forgiving than the export, foreign files not being ours to
choose: `MESH`, `POLYFACE` and loose `3DFACE` entities are all read. The last
two spell out the coordinates of every corner, so a vertex shared by six
triangles arrives six times; those are welded back into shared vertices, and a
face with more than three corners is split into a fan of triangles. Every mesh
in the file is read, so a file holding several bodies comes back as one surface
in several disconnected pieces
* Normals are computed on the way in — the area-weighted mean of the triangles
meeting at each vertex — since a DXF file carries none and `Surface3D` requires
them
* Only the geometry travels. A DXF file has nowhere to put the variables and
metadata a surface carries: `to_zarr` keeps a container whole, and
`as_pyvista` carries the values onto a mesh object
* `ezdxf` is now a dependency
* Surfaces can be assigned to a container's locations, which is what a
surface is usually read in for. `assign_from_surface(surface, name)` records
which side of a sheet each location falls on — above or below a topography, a
seam roof, a weathering front — by comparing it with the sheet's elevation
directly over it, interpolated across the triangle its (x, y) lands in.
`assign_from_solid(solid, name)` asks instead whether a location falls inside
a closed body, an ore envelope or a stope. Both are on every container
* The answer is a metadata column, which is what a domain code should be:
point-wise, carried through `as_data_frame()` and `to_zarr()`, and never seen
by the models. Locations the sheet does not reach are left empty rather than
guessed at
* A grid reads the sheet once for each column of cells rather than once for
each cell — the same (x, y) repeats at every level, and a sheet depends on
nothing else. A `RotatedGrid3D` is not on an axis-aligned lattice and takes
the general path
* A block model can measure the blocks a surface cuts through. The flag
follows the block centre, as a whole-block code does everywhere else, but
naming a `fraction` column as well records the share of each block below the
sheet, or inside the body, over the sub-blocks `discretization` already
defines — what a tonnage near surface needs, where counting a half-buried
block whole is the error
* The mesh arithmetic behind both — splitting faces into triangles, normals
from the triangles meeting at a vertex, counting the edges that belong to one
triangle, interpolating a sheet, testing a body — is in `geoml.geometry`,
taking arrays and returning arrays. `data.py` keeps only what touches a
container, which is the two lines that turn a `Surface3D` into either
* A sheet need not cover everything it is assigned to, and what happens where
it does not is the caller's to choose. Its flag is empty there in any case —
neither above nor below, since the surface says nothing — and the `uncovered`
argument decides the rest: `numpy.nan` by default, so an uncovered block
cannot pass for an empty one in the `fraction` column, `0.0` to count it as
nothing, or `"raise"` to refuse a surface that leaves any location out, which
is the right answer when it was meant to cover the whole model. The footprint
is the triangulation's own, not its convex hull, so a survey boundary, an
L-shaped sheet or a hole around a lake are all respected
* A surface is refused by the assignment it cannot answer: a closed body has
no single elevation above a location, and a sheet has no inside. The two are
told apart by counting the edges that belong to only one triangle — after
welding the vertices, since a mesh can be closed in space while indexing every
triangle's corners separately, as `pyvista.Cylinder` does. A cylinder with one
end open is a sheet by this reckoning, which is the right answer
* A body is also refused when its triangles disagree about which way is out,
because the inside/outside test reads a disagreement as a hole rather than as
an error — quietly, and only where the offending faces are. One wound inwards
throughout is not ambiguous, only reversed, and is turned round rather than
refused; the sign of `geometry.signed_volume` is what tells the two apart, and
its size is the volume the body encloses
* Fixed: `export_contour` raised `TypeError` on every call. It still unpacked
the four arrays `get_contour` once returned, but `get_contour` builds a
`Surface3D` now and hands back that; nothing tested either one, so nothing
noticed

## version 0.5.4
* Printing a variable now says what is on it. `repr` is unchanged — it names
which variable this is — but `str` adds the parts it is made of and the columns
that can be read off it, so they need not be looked up in the source: a
`VectorVariable` lists its components, a categorical one its categories, and
every variable its columns, its quantiles and how many simulations it holds.
Only the names are listed, not whether anything has been written into each
column: knowing that means reading the column, and a column on a large model
lives on disk, which is too much work for a print
* A new `geoml.plots` sub-package, for looking at a data set before modelling
it. `Explorer` holds a container and a choice of variables — one continuous or
vector, one categorical — and answers with figures: `histogram()`, `pairs()`,
`pca(explained=0.9)` and `scene()`. The categorical variable splits and colours
every one of them, which is the question worth asking of most geoscientific
data: does this population behave as one, or as several
* `pca()` draws the components carrying a share of the variance you name, with
the loadings as arrows over the scores. A `CompositionalVariable` is opened up
with the centred log-ratio first — proportions carry a constant sum, so their
covariance is singular and a PCA of them spends a component describing the
closure rather than the data
* `scene()` draws the data where it is, in 1D, 2D or 3D, coloured by a variable
or by one component of a vector variable (`scene(color="Zn")`). For more than a
look at a 3D data set, `as_pyvista()` remains the better road
* The arithmetic behind the figures is in `geoml.plots.prepare`, which draws
nothing and imports no plotting library, so the numbers can be had on their own
and a second way of showing them will read the same functions
* Given a model, `Explorer` reports on it too. `training_curve()` draws the
ELBO with a running mean over it — every value is an estimate, and under
`train_svi` one per *batch*, so the curve is noisy whether or not training has
settled. `transformed_pairs()` passes the measurements through the model's own
warping, already fitted, and asks the two questions a fitted model rests on:
whether the warping made each variable Gaussian, against a normal of the same
mean and spread down the diagonal, and whether what is left is independent,
with the correlation named on every panel. `prediction_scatter()` puts
predicted against measured — one variable with its two distributions along the
sides, where a bias shows that the scatter alone hides, or a panel per
component. `accuracy()` asks whether the simulated spread is the spread the
errors actually have
* `pairs(principal_components=2)` draws the components over the data in the
data's own axes — the reverse of `pca()`, which puts the data on the
components' axes. Each is a line through the mean, one standard deviation of
that component either way, so its length is in the measurements' own units:
long means that direction carries much of the spread, and flat along an axis
means that measurement moves on its own. A line is drawn rather than an arrow
because the sign of a component means nothing, and it is named only where it is
long enough to hang a name on. A `CompositionalVariable` refuses: its
components belong to the log-ratios and are not directions in the proportions
being drawn — `pca()` is the figure for those
* `pairs(upper=...)` fills the upper triangle, which was empty: `"hist2d"` or
`"density"` for where the data sits with the categories pooled, or
`"correlation"` for the coefficient alone, sized by its strength. The lower
half answers which category a point belongs to; the upper answers where the
data as a whole is, which a colour-split scatter hides behind whichever
category was drawn last
* Every figure made of points — `pairs()`, `pca()`, `transformed_pairs()` and
`prediction_scatter()` — takes `kind="hist2d"` for a block model, where the
points are past counting and a scatter is a solid mass whatever its
transparency, `alpha=` for the middling case where transparency is still
enough, and `log_counts=True` to colour the cells by the logarithm of the
count, which is worth having whenever a few cells hold most of the data. Empty
cells are left unpainted rather than drawn as the bottom of the colour scale,
which would fill the panel with a background that looks like data. Counts make
one surface, so a categorical split is pooled rather than drawn five times over
* `transformed_pairs()` numbers its axes `Variable 1`, `Variable 2`, … A
warping may rotate the data or bend it, so a column is generally a mixture of
what was measured rather than any one of it, and naming it after an element
would be wrong in a way nobody would catch
* `pca()` names the variance share to a decimal place — `PC1 (72.2%)` rather
than `PC1 (72%)`
* The figures that compare against observations say so when there is nothing to
compare against. A grid predicted onto carries values everywhere and
measurements nowhere, and `accuracy()`, `histogram()`, `pairs()`, `pca()` and
`transformed_pairs()` now all name that rather than failing somewhere inside
NumPy
* `grade_tonnage()` draws how much material clears each cut-off and how good
it is, on two scales. Without a density it is in volume — or in *area*, for a
two-dimensional grid, which the axis says rather than calling it a volume.
A density may be a number, a metadata column, or a `ContinuousVariable`, and
in that last case its simulations are matched with the grade's **one to one**:
pairing them any other way invents a correlation nobody modelled. Realizations
are carried through separately and drawn as a family with the median over them,
since the curve of the mean model is not the mean of the curves — a cut-off is
a threshold, and averaging either side of it gives different answers
* `grade_tonnage(max_uncertainty=...)` leaves out the blocks the model doubts
most, reading what `Explorer(..., uncertainty=...)` points at. Which column
that is depends on what was modelled — `latent_variance` on a continuous
variable, `uncertainty` on a vector or categorical one, or a column written
alongside — so it is named rather than guessed at. A bare name is looked for on
the variable being drawn and then on the variable *containing* it, which is the
ordinary case rather than an exotic one: grading one component of a vector
variable, the uncertainty that was written belongs to the parent. Failing that,
the metadata. `"Variable.column"` names exactly where to read it, and an array
of one value per location is taken as it comes, which is the way out of every
case a name does not reach. A column that was never filled does not count as
found, so an empty `latent_variance` on a component does not hide the parent's. A block the model cannot
speak for is not tonnage, and counting it flatters the answer at exactly the
cut-offs with least data behind them; the title says how many blocks were kept
* `simulation_pairs()` sets the measurements against the simulations: the
data fills the lower triangle and the simulations the upper, in the same form
and between the same limits, so a simulation that reproduces the data has an
upper half mirroring the lower one. The diagonal carries the measured
histogram with the simulated density drawn over it. Cells rather than points
by default — simulations arrive in location-by-realization blocks running to
millions of values, and a scatter of those is a filled rectangle whatever its
transparency. The two halves are binned according to how many values each
holds, since a few hundred measurements against a hundred thousand simulated
values would otherwise leave the sparse half as scattered single counts
* The simulations are thinned by striding through the block rather than read
whole, one component at a time, so the high-water mark is one component's
simulations and not the whole variable's. The **same stride serves every
component**: thinning them separately would pair the value simulated at one
location with the value simulated at another, and the joint shape — the thing
the figure is for — would be an artefact of the thinning
* `metrics.coverage` and `metrics.goodness` are new, and are plain numbers:
the share of true values falling inside each simulated interval, and Deutsch's
statistic summing that up. Intervals that are too wide count at half the weight
of intervals that are too narrow — claiming a precision the model does not have
is the mistake someone acts on
* The colours are the Explorer's: `palette=` takes a list, or a dict naming
the categories — `{"granite": "#d99"}`, which is what a mapping convention
wants, since a category keeps its colour wherever it lands in the order — and
`cmap=` takes any colormap for the continuous values. The default is `cividis`,
which reads the same to colour-vision-deficient eyes and keeps both its ends
visible against a white background
* Figures are drawn under `geoml.plots.style`, applied through an
`rc_context` to the figure being drawn: importing geoML does not change how
anyone else's plots look. `geoml.plots.PALETTE` is one list, so a rock type
keeps its colour from a histogram to a map; `geoml.plots.use()` takes the
settings globally if you want them
* **matplotlib and plotly are now declared dependencies.** plotly was already
used by `geoml.plotly` and had never been declared. `setup.py` also lists
`geoml.plots`, without which the sub-package would be left out of an install
* `geoml.plots.Interactive` is every one of those figures again, in plotly.
Same class, same method names, same arguments — `histogram`, `pairs`, `pca`,
`scene`, `training_curve`, `transformed_pairs`, `simulation_pairs`,
`prediction_scatter`, `accuracy`, `grade_tonnage` — differing only in taking a
size in pixels rather than in inches. `Explorer` is the one to save and print;
`Interactive` is the one to look at, and what it adds is what a printed figure
cannot do: zooming a panel of a scatter matrix takes its whole row and column
with it, so a cluster can be followed across every pair at once; clicking a
category in the legend takes it out of every panel; and a 3D scene can be
turned around
* Both classes now share `plots.base.Selection`, which holds the container,
the choice of variables and the colours, and settles what a figure is *about*
before anything is drawn. The numbers still come from `plots.prepare` — read
by both, so the two backends cannot drift into showing different things
* `geoml.plots.Dashboard` puts several figures on one page and **links their
selections**: drag a box or a lasso over any panel and the same locations light
up in every other one, which is the question no single figure answers — those
samples out on the tail of the histogram, where are they on the map? Double-
click to clear. Every trace `Interactive` draws from locations carries, in
`customdata`, the row of the container each point came from, and selections are
matched on those rows rather than on position, so a sample may be a bar in one
panel and a dot in another and still be the same sample
* What carries no rows is left alone, which is the honest answer rather than a
guess: a counted cell holds a number of points and not one of them, a simulated
value is a realization and not a place, and a 3D scene has neither a box to drag
over it nor per-point selection to show a brush made elsewhere
* `scene(clip=[0, 0.99])`, on both `Explorer` and `Interactive`, ends the
colour scale at a pair of quantiles rather than at the extremes. One value far
from the rest otherwise takes the whole scale with it and leaves every other
point in a single shade, which is the usual fate of an assay: a Cd with one
value twenty-five times the next draws as a uniformly dark map, and the same
map clipped at the 99th percentile shows the structure that was there all
along. Nothing is dropped and nothing is altered — the points past the end take
the end colour, and a hover still reports what was measured. `[0.01, 0.99]`
takes both tails; a menu clips each of its choices by its own quantiles
* A categorical variable with no category measured anywhere now says so.
Every series is drawn per category, so none of them meant no traces at all --
a blank figure and no complaint. The commonest way to arrive there is passing
the values to `add_categorical_variable` where the *categories* belong, and
the message says as much
* The dashboard's linked selection got out of its own way. Plotly redraws
every trace of a figure whatever changed, at about a millisecond each, so a
selection used to freeze the page for the sum over every panel at once. Now a
panel drawn from no locations — a training curve, a grade-tonnage — is skipped
rather than redrawn to no effect; a panel already clear is not cleared again;
the panels are done lightest first, so the map answers at once and the scatter
matrix follows; and the browser is handed back between them, so each appears
as it is finished instead of all of them after a frozen half-second. A
selection arriving mid-walk replaces the pending one rather than being dropped.
What is left is the cost of the heaviest single panel, which is plotly's to
pay: a seven-by-seven matrix split five ways is 161 traces and about 185ms of
redraw. Fewer categories, fewer components, or `kind="hist2d"` are the levers
on that
* `Interactive.scene(color=[...])` puts a menu on the figure instead of
drawing one variable, one entry per choice — and `color=["Elements"]` names
every component of a vector variable, which is the answer to the error that
says a vector is not one colour scale. The ground is drawn once and the values
are swapped over it rather than a trace being carried per choice: lighter, a
block model's coordinates being the bulk of it, and truer to what is being
asked, since the cloud stays where it is while the variable over it changes.
Only the locations measured in all of them are drawn, for that same reason. In
1D, where the value is the vertical axis and carries no colour, the menu moves
the points rather than repainting them
* 3D scenes are linked the other way about. There is no brushing them, but
there is turning them, and a page of several answers a question one of them
cannot: the same body of ground under two variables, seen from the same angle.
Turn one and the rest follow. The exchange ends by each panel ignoring a view
it already holds — plotly answers a camera move with a camera event of its own,
which arrives too late for a flag to catch, so the panels would otherwise hand
the view back and forth for ever
* The rows of a dashboard need not be equal. Nest an entry in `figures` and it
becomes a row of its own, however many panels are in it — a map across the full
width, a histogram and a training curve side by side beneath it, the scatter
matrix under those. Nest nothing and the figures are dealt out `columns` to a
row as before. A `(caption, figure)` pair is still a captioned figure and not a
row of two: both arrive as a tuple, and what tells them apart is that a caption
is a name and a figure is not
* `board.write_html("jura.html")` writes a page that carries plotly with it —
some four megabytes — so it opens anywhere, with no network and no geoML.
`plotlyjs="cdn"` fetches the library instead, for a file small enough to keep
in a repository. `eda.dashboard()` is the short way, naming the figures to
draw; `Dashboard([...])` takes figures instead of names, drawn with whatever
arguments you like and captioned where the caption is a better title
* A dashboard renders in a notebook, inside an iframe. A notebook is not
obliged to run a script it is handed and JupyterLab does not, which is how a
bare `<script>` works in one notebook and silently does nothing in the next; an
iframe is a document in its own right and runs its own, everywhere. Panels keep
the proportions they were drawn at rather than being stretched to their column
— a scatter matrix poured into half a page turns every panel into a slot
* `plots.style` grew `TEMPLATE`, the same look for plotly, as a plain dict:
reading the package's colours costs no import of plotly, the same way
`geoml.plotly` builds a figure without importing it
* `plots.prepare` gained the arithmetic both backends now share —
`counts_2d`, `density_grid`, `density_curve`, `normal_curve` and `cells` — and
`prediction_values` also returns which rows it drew, without which a
predicted-against-measured panel could not say who its outliers are

* A model can be drawn: `model.to_dot()` writes the whole thing as a Graphviz
diagram — the coordinates that go in, the latent network, the warpings on the
way out and the variables that come out — and `network.to_dot()` draws a latent
network on its own. Boxes are coloured by the part they play, a `Stack` or
`Concatenate` is the circle the hand-drawn diagrams use, and every arrow is
labelled with the number of variables travelling along it. The warpings are
drawn the way the model *generates* a value rather than the way it reads one,
so all the arrows run together. Render with `dot -Tpng network.dot -o
network.png`
* The new `geoml.graphviz` module imports nothing, the way `geoml.plotly`
builds a figure without importing plotly: Graphviz is needed to look at a
diagram, never to write one. Its `PALETTE` is a plain dict, so the colours are
yours to change
* A diagram says one thing the printed tree cannot: a node feeding several
branches is drawn once, with an arrow to each. `str` has to repeat it, which
reads as though there were two
* Latent nodes have names. Every node in `latent.py` takes an optional `name`,
and one built without it is numbered against the nodes it is connected to —
`BasicGP_1`, `BasicGP_2` — so the branches of a network can be told apart in
`str(network)` without naming anything. The numbering looks along the network
in both directions, not only upwards, because a new node's siblings are not
among its ancestors; two subnetworks built separately and joined only later are
the one case it cannot see through, and `get_node` says so if it is ever asked
for a name they share
* `network.get_node(name)` returns a node by name, among the top node and
everything feeding it. Names given by the user are recorded as constructor
arguments, so they are saved with the model; automatic ones are regenerated by
the replay, in the same order, and come back identical
* The node incompatibility errors name the nodes involved. A size mismatch
used to read `All parents must have the same size. Found [1, 2]`, and now reads
`Add_1: all parents must have the same size. Found BasicGP_1 (size 1),
BasicGP_2 (size 2)`; a GP built on a parent that cannot propagate inducing
points names both

## version 0.5.3
* Point-wise information can be attached to any container with
`obj.add_metadata(name, values)` and read back with `obj.get_metadata(name)` —
an air/solid code, a cross-validation fold, a sample weight: things known per
location that the models never see. It follows the object through subsetting,
`as_data_frame()` and now `to_zarr()`/`open()`, which used to drop it silently
* **Breaking:** `obj.metadata` is a dict of columns, not a `DataFrame`. A
column is the same `_Attribute` a variable is built from, so it spills to Zarr
when large and carries the same helpers — `grid.metadata["air"].as_cube()` and
`fill_pyvista_blocks()` work on it. Text is kept as integer codes plus labels:
as an object column a rock code on a two-million-block model costs 105 MB
against 1.9 MB as codes, and object arrays can never spill to disk. Code that
read `obj.metadata["fold"]` expecting a column of values wants
`obj.get_metadata("fold")`
* `_Attribute` is a module-level class instead of one nested inside
`_Variable`, and it is where the text-as-codes handling lives
(`_Attribute.encoded`). Two fixes came with it: an attribute built from values
now chooses its backend by size like an empty one does, instead of being pinned
in RAM whatever its length; and reopening one from disk no longer allocates a
throwaway array first
* **Breaking:** the categorical variables hold codes too.
`RockTypeVariable.predicted` / `measurements_a` / `measurements_b`,
`BinaryVariable.measurements` / `predicted` and their subclasses were arrays of
Python strings — a predicted rock type on a 27-million-block model is about
1.4 GB that cannot be spilled to disk, against 27 MB as `int8`. Read them with
`attribute.to_numpy()`, which gives the labels back; `attribute.values` is now
the codes. Anything that writes a prediction writes the position of the winning
label. -1 means "not measured here" and reads back as the empty string, as it
did before. A measurement column carries **its own** categories, not the
variable's, so an `AnomalyVariable` measured as "hit"/"none" still reports
"none" even though the variable calls that class `_dummy`. Containers saved by
an earlier version are re-encoded when they are opened
* Slicing a categorical variable now drops the categories that are no longer
present, all the way through: the components go with them, and the prediction
is re-coded to the shorter list, `update` writing the winning label's position.
Before, only `labels` was reset, leaving a variable that claimed one category
and carried three. An `OrderedRockType`'s implicit values are recomputed rather
than carried over — they are positions in the label sequence, so dropping a
category shifts every one after it, and a slice would otherwise be trained
against a scale that no longer existed. A slice with nothing measured in it
keeps the categories it was declared with
* Categories that are not text are left as they are. Deriving them stringified
them, so a variable measured as `1`/`2` compared its predictions against `'1'`
and `'2'` and every metric came out at chance. They are stringified only on the
way to disk, as the variable's own labels always have been — which also fixes a
crash, older than this release, that made a variable with integer categories
impossible to save at all
* Interval columns can be renamed with `holes.rename(table, {"Pb_pct_ICP":
"Pb"})`, or `table.rename(...)` on the table itself. The roles travel with the
columns: renaming the data frame directly left them behind, keyed by the old
name, so the column quietly stopped being a grade and printing or compositing
the table raised a `KeyError`. Renaming the frame before it is added still
works as it always did — this is for a database that was built for you, by
`datasets.macpass()` or by compositing, where the frames were never in reach.
Column names become variable names in `as_point_data()`, so it is also the last
chance to tidy them before they reach a model

## version 0.5.2
* Simulated predictions are reproducible. The likelihood noise was drawn from
TensorFlow's global generator with a hard-coded seed, so `options.seed` never
reached it and two predictions of the same model gave different simulations.
The draws are now stateless and seeded from `options.seed`, in a stream of
their own so they do not follow the latent draws
* The noise on discretized blocks is drawn per sub-block position: the `k`
sub-blocks of a block get independent values, and every block gets the same `k`
values. Averaging them is what integrates the noise over the block, and the
shared pattern means the noise shifts the map rather than roughening it, which
is what makes simulations of a block model look like the field and not like
static. Since the draw is indexed by sub-block rather than by row, simulations
no longer depend on how the prediction was batched
* `prediction_batch_size` now counts the rows the model actually evaluates. A
block with a [5, 5, 5] discretization fans out into 125 of them, so the default
20000 handed the model 2.5 million rows at once and had to be shrunk by hand to
avoid running out of memory; the setting now means the same thing for every
container
* The block fan-out is built by broadcasting instead of a Python loop over
blocks — 8x faster on a 40000-block model — and the noise is broadcast over the
blocks instead of being tiled into a second copy of the largest tensor in the
prediction
* The two `white_noise` implementations, which disagreed on what "coherent"
meant, are now one on the base likelihood

* **Breaking:** `as_pyvista()` and the `fill_pyvista_*` methods no longer export
simulations by default. Each one is a full-length array in the exported object,
so a block model with twenty of them carried twenty copies of itself into
memory whether or not they were going to be looked at. Pass `simulations=True`
for all of them, an `int` for the first n, or a sequence of indices —
`grid.as_pyvista(simulations=[0, 5])`
* The simulations that are exported are now read in a single pass. The store is
chunked along rows only, so a chunk holds every simulation of a row band and
reading one column at a time decompressed the whole array once per simulation.
On a 4 million point grid with 20 simulations the read went from 2.9 s to 1.5 s,
and the default export — which no longer touches the simulations at all — from
2.9 s to 0.02 s
* `as_data_frame()` read its simulations the same way, one column at a time,
and now takes the same single pass. Its `simulations` argument also accepts an
`int` or a sequence of indices, not only a flag, so a data frame can carry a
couple of realizations instead of all or nothing. Its defaults are unchanged
* The check that skips an empty attribute is vectorized. It used the built-in
`all()` over a NumPy array, walking it one element at a time in Python, for
every attribute of every export

## version 0.5.1
* A model can now be reproduced from a single call. Parameters are initialized
at random when an object is built, which happens before any model exists to
carry `options.seed`, so the draws now come from one generator held in the new
`random` module: call `geoml.set_seed(1234)` before building and the same
objects start from the same values, in this run or the next. Left alone, the
generator is seeded by the operating system, as before. Scripts that pinned a
build with `np.random.seed()` must call `geoml.set_seed()` instead —
initialization no longer reads NumPy's or TensorFlow's global generators
* Objects now print themselves. `repr` identifies one on a single line, as the
call that would build it — `Covariance(Gaussian(), Isotropic(range=111.0))` —
reading the arguments each object already records for persistence, and showing
the *current* value of a parameter rather than the one it was built with, since
training moves it. Objects holding many or large parameters, such as a GP node,
fall back to naming their configuration. Data containers, variables and models
keep summarizing themselves as they did. This also fixes `repr` raising
`NotImplementedError` for every kernel and every likelihood, which took
debuggers and failing assertions down with it
* `str` lays out an object's state: its parameter values, and everything
registered inside it, indented. A latent network therefore reads as its
composition, from the output node down through its parents to the input, with
each node's size. Large parameter arrays are summarized by shape and range
instead of being printed in full. The four near-identical `pretty_print`
implementations in `kernels` and `transform` are now one, inherited from
`Parametric`

* geoML no longer disturbs the generators the caller is using: building a
`RandomProjections` transform used to reset NumPy's global seed outright, and
training did the same before shuffling its batches. Both now draw from a
generator of their own, and remain reproducible from the seed they are given

## version 0.5.0
* `DrillholeData` was rewritten from scratch, in a new `drillhole` module.
It is now built from a collar and an optional survey table, and interval files
(assay, lithology, and any others) are added to it separately, each as an
`IntervalTable`. Hole paths are desurveyed by minimum curvature, so deviated
holes are positioned properly; holes with no survey are taken as straight,
along the collar attitude. Databases that record a downward hole with a
negative dip are supported through `dip_positive_down=False`
* Interval tables declare the role of each column — grade, categorical,
density, flag or ignore — which decides how it is composited. Roles can be
redeclared afterwards
* Interval data is checked on entry: overlapping intervals, non-positive
lengths, missing depths and holes with no collar are reported as errors, and
gaps and intervals running past the end of the hole as warnings, with the
policy chosen by `on_error`
* New compositing methods, all of which return a new object with every table on
one shared support: `composite()` (fixed length, honouring the boundaries of a
named categorical domain), `composite_fixed()` and `composite_to()` (onto the
intervals of another table). Numeric columns are averaged weighted by length,
or by length times density where a density column is declared, skipping
missing values; categories are decided by the greatest length in the composite
* `as_point_data()` converts the interval data to points, merging the tables
row for row. Compositional and vector variables are assembled here, with the
processing raw assays need, in order: each column is divided by its unit
("%", "ppm", "g/t", …, or a number), which has to be declared and is what
lets parts measured in different units be added together; a sample missing any
one part is marked missing entirely, since the parts only carry information
relative to each other; non-positive parts are replaced by half the smallest
positive value of their own column, the usual substitution for values below
detection, which a log-ratio transform cannot otherwise take; and a rest part
is added, holding whatever is left of the whole, so a barren sample is nearly
all rest and a rich one much less so. Where the parts leave no room, the rest
is held at the smallest part found and the sample is scaled down to fit, with
a warning — which usually means a unit was declared wrongly
* The database can be cut down before it is used: `subset_region()` keeps the
holes whose collar falls within a bounding box, `subset_holes()` those named
explicitly, and `filter_intervals()` the intervals holding a given value — or,
with `whole_holes`, every hole in which that value was logged
* Logging codes can be lumped into modelling domains with
`group_categories()`, which takes a mapping from each new category to the codes
it takes in, and reports the codes left out. Since there is no GUI to click
them together, `category_legend()` lists the distinct codes with the number of
intervals and the metreage each accounts for, sorted by length and with a
`group` column ready to be edited: write it to a spreadsheet, fill the column
in, and read it straight back into `group_categories()`. Intervals carrying no
value get a row of their own, so the legend accounts for the whole table
* `fill_unlogged()` gives a category to the ground a log does not account for,
which is what a database holds when only the intervals of interest were
logged. Intervals that exist but carry no value take the new label, and the
depths no interval covers become intervals of their own, reaching from the
collar to the bottom of the hole — so a hole nobody logged comes out labelled
end to end, and the log covers every hole continuously afterwards
* `as_classification_input()` still produces points of zero effective support,
including the contacts between rock types carrying both neighbouring classes
* New `datasets.macpass()` loader for the Macmillan Pass drillhole database,
published by Fireweed Metals Corp. The data is not distributed with geoML; see
the docstring for where to get it and the terms that apply
* Memory-efficient data storage: variable arrays are now backed by NumPy in RAM
or chunked Zarr on disk, chosen automatically by size, so large projects no
longer need to hold every array in memory
* Simulations are stored as a single `(n_data, n_sim)` array (a dedicated
dimension) rather than a list of separate arrays
* New `variable.simulation(i)` and `variable.n_sim`: a single simulation can
again be handled as an attribute, with the usual `as_image()`, `as_cube()`,
`smooth()` and contouring helpers
* Grid and block coordinates are now generated lazily: a regular grid no longer
holds its full `(n_data, n_dim)` coordinate array in memory, but produces the
requested rows on demand, so very large grids can be built and predicted into
without the coordinates dominating RAM
* Point coordinates and `GaussianData`'s input variance are stored the same way
as variable arrays (NumPy in RAM or Zarr on disk, by size), so a large point
cloud no longer has to stay in memory
* Fixed a bug where the input variance was rebuilt for the whole object on every
prediction batch, making prediction quadratic in the number of points; it is now
generated for the requested batch only
* Fixed `GaussianData`: predicting into one no longer fails, subsetting keeps the
object's class and variance instead of degrading to `PointData`, and the variance
is included in `as_data_frame()`
* Containers can be persisted to and reloaded from a single Zarr store with
`to_zarr()` / `open()`, covering all point-based containers and variable types
* Trained models can be saved and loaded whole with `model.save()` and
`Model.open()`, in a new `persistence` module: structure, trained parameters
and training data are all kept, so a reloaded model predicts exactly as it did
and can be trained further. It remembers its variables and their types, and so
creates them on any new data object it predicts on. `Model.open(path, data=...)`
rebuilds the same model around a different data set
* Quantiles and probabilities are computed lazily in a single pass over the
simulations, without materializing them
* `reset_probabilities()` is now the inverse of `reset_quantiles()`: it takes
cutoff values and returns cumulative probabilities in (0, 1)
* Fixed prediction of categorical variables, which failed with a `TypeError`
after the prediction posterior started being cached between batches
* Fixed variables being degraded to a more basic type when built from an
existing one, either by predicting on a new data object or through
`from_data_frame()`: a categorical variable became a rock type variable, an
ordered rock type became an unordered one, and an anomaly became a plain binary
variable. The same applied to data objects, where `Blocks2D.from_bounding_box()`
and its relatives returned a plain grid
* `CategoricalVariable` can now be built from a data frame with a single
measurements column, instead of the two contact columns a rock type needs
* Fixed the `n_sim` argument being ignored when predicting with the standard
`GP` model, which failed for any value other than 50
* Fixed construction of `OrderedRockType` and rock-type/binary variables from
string measurements
* Fixed a bug where a parameter shared between components (e.g. a transform
reused by several kernels) was counted more than once by `get_parameter_values`
and `save_state`
* New dependencies: `zarr`, `xarray`, `dask`

## version 0.4.1
* Support for predictions over blocks with discretization
* Delta method for change of support

## version 0.4.0
* Implemented multivariate data and likelihoods
* Warpings are now multivariate
* New warpings: `PCA`, `RobustPCA`, `Rotation`, and `ContinuousNormalizingFlow`
* Streamlined code to deal with compositional data
* Streamlined quantiles and percentiles computation, now directly from simulations
* Added option to smooth data when converting it to data to image or cube format
* Added convenience methods to compute model metrics


## version 0.3.5
* Support for rotated 3D grids
* Simulations now have the option to include the likelihood variance
* Improvements and bug fixes
* Product of Experts is now the default for GP latent variables

## version 0.3.4
* Improved pyvista integration
* Multiscale modeling did not work
* Compatibility with TensorFlow v2.11

## version 0.3.3
* Improvements and optimizations
* Heteroscedastic likelihoods
* A likelihood for conformable layers
* A likelihood for limited time-aware implicit modeling
* Ability to use a random field to move data points
* Ellipsoidal trends
* Limited support for faults
* (Experimental) multiscale modeling (refinement with increasingly denser inducing point sets)

## version 0.3.2
* Tensorized latent variable removed for now (back to the old chalkboard).
* Full deep GP implementation.
* Added Jura dataset.
* Added likelihoods for compositional variables.
* `kernels` module now has a specific object for covariance functions.
* Implemented the ability to convert data objects to `pyvista`.
* Implemented `Surface3D` data object.

## version 0.3.1
* New autoregressive latent variable.
* New tensorized latent variable.
* New Product of Experts latent variable.
* Spline warping is now centered at zero.
* Ensembles now handled in a specific model.

## version 0.3.0
* Introduced variational models.
* All training is now gradient-based.
* Changed the data manipulation system, defining variable types (continuous,
categorical, etc.) and Attributes (mean, variance, entropy, etc.) that can be
accessed and plotted directly.
* Plotly interface is now in a specific module.
* New `Section3D` data object.

## version 0.2.0
* Code updated to TensorFlow 2.0.
* Cubic convolution implemented in TensorFlow.
* Simplified kernel API.
* New auxiliary functions in `tftools` module.
* Cubic splines implemented in TensorFlow.
* Added Cerro do Andrade dataset.
* Notebooks moved to Google Colaboratory.
* Training is now based on Particle Swarm Optimization.
* Included a `GPOptions` class to control verbosity, batch size and
other model options.
* The covariance matrix for directional data is now computed with a 
finite difference method.
* Drillholes can now be segmented in roughly fixed intervals.

## version 0.1.3
* Introduced parallel sparse cubic convolution interpolation.
* Introduced the sparse GP model.
* Introduced the SPICE model for scalability.
* New auxiliary functions in `tftools` module, such as 
conjugate gradient solver and Lanczos decomposition.
* Limited support for modeling non-spatial data.

## version 0.1.2
* Introduced the `jitter` parameter in the models, for improved 
numerical stability.
* Corrected a bug where setting the value of a parameter beyond its limits
would not extend the limits.
* Implemented the `prod_n()` and `safe_logdet()` functions in the 
`tftools` module.
* Implemented the `aggregate_categorical()` method for point data methods.
* Optimized the batch prediction process.
* Implemented saving and loading for model parameters.
* Implemented the neural network transform.
* Now multiple transforms can be chained and/or shared between multiple
kernels.
* The classification algorithms now output an uncertainty metric, weighting
the entropy and predictive variance.

## version 0.1.1
* Added support for products of kernels.
* Fixed bug in `geoml.tftools.prod_n()` function.
* `geoml.kernels.Gaussian.covariance_matrix(x, y)` now
adds some jitter to the covariance matrix when x == y, to
avoid Cholesky decomposition problems.
* `geoml.warping.Softplus` now corrects non-positive values
in its input.
* Corrected a bug where the wrong nugget would be added
while making a prediction.
* Added sunspot number data and example notebook.
* Implemented cubic convolution fully in TensorFlow.
* Set the default `n_knots=5` in `geoml.warping.CubicSpline.__init__()`.