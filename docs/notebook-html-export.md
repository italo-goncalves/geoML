# Exporting a notebook to HTML with working pyvista scenes

A Jupyter notebook full of pyvista scenes can be converted to a single HTML
file in which every scene stays fully interactive — rotate, zoom, pan — in
any browser, with no Python behind it. The catch is the Jupyter backend.

## The trap, and the backend that avoids it

PyVista's default notebook backend (`'trame'`) is a **live widget**: the
scene in the cell is served by the running kernel, so `jupyter nbconvert`
captures a dead placeholder. This is what makes the export look impossible.

The `'html'` backend is the one built for this: it serializes each scene's
geometry into a self-contained vtk.js viewer *inside the cell output*, so
the scene needs no kernel ever again. Available in pyvista ≥ 0.44
(verified here on 0.47.3).

## Recipe

```python
import pyvista as pv
pv.set_jupyter_backend('html')   # once, at the top of the notebook
```

Show plotters as usual, run the notebook top to bottom, then:

```bash
jupyter nbconvert --to html notebook.ipynb
```

To embed only selected scenes, set it per call instead:
`pl.show(jupyter_backend='html')`. For a standalone file per scene, outside
any notebook, there is `plotter.export_html("scene.html")`.

## Caveats

1. **"Interactive" means the camera.** Anything backed by Python — slider
   widgets, picking callbacks, live updates — dies with the kernel by
   definition. Surface meshes are the well-trodden path in vtk.js; volume
   rendering support is partial.

2. **File size is the real constraint.** The geometry is embedded in the
   HTML, so a notebook of block-model contours and topographies can
   produce a file of hundreds of MB that browsers choke on.
   `simplify(max_error)` exists for exactly this:

   ```python
   shell = bm.get_contour("assay/Ag/prediction", cutoff,
                          close="above", simplify=1.0)
   below = topo.simplify(1.0).clip_meshes([shell])
   ```

   keeps each scene a few MB at sub-metre fidelity (the argument is the
   largest distance the simplified surface may move, in coordinate units).

3. **The export itself renders.** Run the conversion from an environment
   where pyvista renders off-screen — under WSL that means the conda env
   with the libstdc++ fix installed (see `wsl-pyvista-rendering.md`).

## One report, no server

`plots.Dashboard` already writes self-contained HTML for the plotly
figures, linked selection included. The `'html'` backend is the pyvista
counterpart: a complete deliverable — dashboard plus interactive 3D
scenes — can ship as plain HTML files that open anywhere.
