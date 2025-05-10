
# Joint PDF and ZND Plotting Tools for Detonation Simulations

This repository provides Python scripts for post-processing and visualization of detonation simulation data, particularly focused on plotting 1D slices, computing joint probability density functions (joint PDFs), and overlaying theoretical ZND (Zel'dovich–von Neumann–Döring) profiles.

The scripts are based on and extend the `plot_line.py` utility from the [Athena++](https://github.com/PrincetonUniversity/athena-public-version) framework.

## Scripts Included

### `plot_lines_modifide_with_znd.py`
- Plots simulation data from `.athdf`, `.hst`, or `.tab` files.
- Computes and overlays a theoretical ZND solution.
- Generates joint PDFs (e.g., Pressure vs Temperature).
- Includes CJ point predictions based on `Q` and `G` values extracted from the profile name. The values `Q` and `G` corespond to the normilzed heat and gamma (heat capacity) used.

### `plot_lines_modifide_for_cluster.py`
- Similar to the above, adapted for execution on a cluster environment with paths suited for remote storage.
- Includes support for plotting multiple Mach number bounds and a more compact plotting configuration.

### `joint_pdf.py`
- Computes and visualizes a 2D joint PDF of any two simulation fields (e.g., Temperature vs Density) from volumetric Athena++ output.
- Supports direction slicing and axis limits.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- Athena++'s `athena_read.py` utility must be available in your Python path.

## Example Usage

```bash
python plot_lines_modifide_with_znd.py data.athdf Pressure Temperature output_name \
  --x_label "P/P₀" --y_label "T/T₀" --profile_name Q_25_G_1.4
```

```bash
python joint_pdf.py data.athdf Pressure Temperature output_plot \
  -d 3 --x_label "P" --y_label "T" --x_min 100 --x_max 200
```

## Based on Athena++ Plot Tools

These scripts are adapted from `plot_line.py` used by the [Athena++ code](https://www.astro.princeton.edu/~jstone/Athena++/), a high-performance astrophysical hydrodynamics code.

> Athena++ plotting documentation: [Athena++ official documentation](https://www.astro.princeton.edu/~jstone/Athena++/)

## File Structure Expectations

- Simulation output files should be in `.athdf`, `.hst`, or `.tab` format.
- ZND profiles should be located under the expected relative or absolute paths used in the scripts (modify as needed).
- Profile names must include both `Q_` and `G_` with values to extract physical parameters.

## Output

Each script generates:
- A PNG image showing the joint PDF (log-scaled) of the selected variables.
- Overlaid ZND profiles or CJ lines for comparison.
