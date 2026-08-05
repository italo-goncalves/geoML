# Fixing pyvista/VTK rendering in a conda env under WSL

A conda environment that is a few years old will fail to render anything with
pyvista under WSL, even though WSLg itself works and `import pyvista` succeeds.
This note explains how to recognise the problem, confirm it, and fix it. The
commands are generic — no machine-specific paths — so they apply to any machine.

## Symptom

Any render attempt fails. VTK tries GLX, then EGL, then OSMesa, and the process
usually dies with a segmentation fault (exit code 139):

```
vtkXOpenGLRenderWindow: Could not find a decent config
vtkXOpenGLRenderWindow: Could not find a decent visual
vtkEGLRenderWindow: Could not initialize a device. Exiting...
vtkOSOpenGLRenderWindow: libOSMesa not found.
Segmentation fault (core dumped)
```

This affects off-screen rendering and interactive windows alike. It typically
appears on an environment that *used to* work and was not changed — the host
distribution moved forward while the environment stayed still.

## Cause

Mesa's software rasteriser, `llvmpipe`, is what actually draws under WSLg. Newer
Mesa builds it against LLVM 20, and `libLLVM-20.so` requires `GLIBCXX_3.4.32`.

An old conda environment ships `libstdcxx-ng 11.2`, i.e. `libstdc++.so.6.0.29`,
which provides only up to `GLIBCXX_3.4.29`. Because the environment's `lib`
directory takes precedence, that old library shadows the system one inside the
Python process. `libLLVM-20.so` then fails to load, `llvmpipe` never registers,
no GL framebuffer configurations exist, and VTK falls through every backend.

It is *not* an X11, WSLg or driver problem, and it is not a pyvista bug.

## Confirm it

Anything below that prints a number lower than the LLVM requirement confirms
the diagnosis. Run with the environment activated:

```bash
echo "conda provides : $(strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -V | tail -1)"
echo "system provides: $(strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -V | tail -1)"
echo "LLVM needs     : $(objdump -p /usr/lib/x86_64-linux-gnu/libLLVM-*.so | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -V | tail -1)"
```

A second, independent check — the first command should fail and the second
should report `llvmpipe`, which proves the environment's libraries are the cause:

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" glxinfo -B | grep -E 'Device|Error'
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 glxinfo -B | grep -E 'Device|Error'
```

`glxinfo` comes from `mesa-utils` (`sudo apt install mesa-utils`); it is a
convenience for diagnosis only and is not needed for the fix.

The paths above assume a Debian/Ubuntu layout. If they do not exist, ask the
linker where the library actually is — the fix below does this automatically:

```bash
ldconfig -p | grep 'libstdc++.so.6 '
```

## Fix

Preload the system `libstdc++` so it wins over the environment's copy. This is
safe rather than a hack: the system library is a strict superset of the old one,
so every conda binary keeps working.

Making it a conda activation hook means it applies automatically on activation
and is removed on deactivation, so `LD_PRELOAD` never leaks into other shells.
**Activate the target environment first**, then paste the whole block:

```bash
set -e
SYS_LIBSTDCXX=$(ldconfig -p | grep 'libstdc++.so.6 ' | grep -v -i conda | awk '{print $NF}' | head -1)
test -n "$SYS_LIBSTDCXX" || { echo "no system libstdc++ found"; exit 1; }
echo "using $SYS_LIBSTDCXX"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh" <<EOF
# Mesa's software rasteriser (llvmpipe) loads libLLVM-20.so, which needs
# GLIBCXX_3.4.32. This env's libstdcxx-ng only provides 3.4.29, so the driver
# fails to load and VTK/pyvista segfault after falling through GLX, EGL and
# OSMesa. Preload the system libstdc++, which is a strict superset.
_sys_libstdcxx=$SYS_LIBSTDCXX
if [ -e "\$_sys_libstdcxx" ]; then
    _OLD_LD_PRELOAD="\${LD_PRELOAD-}"; export _OLD_LD_PRELOAD
    _LD_PRELOAD_SET=1; export _LD_PRELOAD_SET
    if [ -n "\${LD_PRELOAD-}" ]; then
        export LD_PRELOAD="\$_sys_libstdcxx:\$LD_PRELOAD"
    else
        export LD_PRELOAD="\$_sys_libstdcxx"
    fi
fi
unset _sys_libstdcxx
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh" <<'EOF'
# Undo the LD_PRELOAD set by activate.d/libstdcxx.sh
if [ -n "${_LD_PRELOAD_SET-}" ]; then
    if [ -n "${_OLD_LD_PRELOAD-}" ]; then
        export LD_PRELOAD="$_OLD_LD_PRELOAD"
    else
        unset LD_PRELOAD
    fi
    unset _OLD_LD_PRELOAD _LD_PRELOAD_SET
fi
EOF

chmod +x "$CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh" \
         "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh"
echo "installed"
```

Note the two heredocs differ deliberately: the first is unquoted so
`$SYS_LIBSTDCXX` is baked in at install time, with every other `$` escaped; the
second is quoted so nothing expands.

## Verify

Open a **new** shell — the hook only runs on activation — and check that the
variable appears, that a render succeeds, and that deactivating cleans up:

```bash
echo "before: [$LD_PRELOAD]"
conda activate <env>
echo "after : [$LD_PRELOAD]"
python -c "import pyvista as pv; p=pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); print('RENDER OK', p.screenshot(return_img=True).shape)"
conda deactivate
echo "clean : [$LD_PRELOAD]"
```

Expected: empty, then the library path, then `RENDER OK (...)`, then empty again.

## Undo

```bash
rm "$CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh" \
   "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh"
```

## Alternative: upgrade the library instead

The root fix is to bring the environment's `libstdcxx-ng` up to date, after
which no preload is needed:

```bash
conda install -c conda-forge 'libstdcxx-ng>=13'
```

Check what the solver proposes before agreeing. On an Anaconda-based
environment this can pull in unrelated packages and downgrade the `anaconda`
metapackage, which is why the activation hook is usually the lighter option.
Add `--dry-run` to see the plan without applying it.

## Notes

- `LD_PRELOAD` applies to every process started from an activated shell, not
  just Python. That is harmless here, but it is the reason for the deactivation
  hook rather than an `export` in `~/.bashrc`.
- Upgrading `libstdcxx-ng` later makes the hook redundant, not harmful.
- If rendering breaks again after a `conda update`, check that both files
  survived.
- Verify WSLg itself is healthy before blaming the environment: `echo $DISPLAY`
  should print `:0` and `glxinfo -B` run *without* conda on the path should
  report `llvmpipe`. Beware that in PowerShell, `wsl.exe -e bash -c "... \$VAR"`
  does not escape the variable — PowerShell expands it first and it arrives
  empty, which looks like a broken WSLg but is not.
