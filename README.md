# Munich bus-network fragility analysis

This repository contains the Python code used to reproduce the computational
results reported in the paper on efficiency metrics under disruptions and
delays.

The analysis builds two graph representations of the Munich bus network:

- a service-level directed multigraph;
- an aggregated stop-to-stop directed graph.

For each graph, the script computes the flow-weighted backup fragility
measures, single-arc coverage, delay elasticity, the greedy top-10 coverage
selection, and line-level criticality. It also evaluates the literature
measures used for comparison in the paper: node and edge betweenness, node and
edge information centrality, network traffic vulnerability, and agglomeration
vulnerability.

## Requirements

- Python 3.11 or a later compatible version
- Internet access when downloading the GTFS archive or OpenStreetMap background

## Installation

Clone or download this repository, open a terminal in its root directory, and
create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the analysis

```bash
python reproduce_paper_results.py
```

The results are written to the automatically created `results/` directory.
They include node, edge, line, and top-10 tables; Spearman correlation
matrices; canonical-path transfer statistics; figures; and a computation-time
report.

## Input data and reproducibility

By default, the script downloads the current MVG GTFS archive and selects the
first valid weekday contained in it. Because the online GTFS archive changes
over time, this default is convenient for testing but does not guarantee that
future runs reproduce the exact paper results.

For a reproducible paper run:

1. Retain the exact GTFS ZIP archive used for the experiments.
2. Set `GTFS_SOURCE` to the local path of that archive.
3. Set `ANALYSIS_DATE` explicitly in `YYYY-MM-DD` format.
4. Record the data source, download date, and selected analysis date.

The main configuration parameters are located near the beginning of
`reproduce_paper_results.py`.

## Walking sensitivity analysis

The baseline excludes walking connectors. To run the additional sensitivity
analysis with walking connections between stops, set:

```python
RUN_WALKING_SENSITIVITY = True
```

The default sensitivity settings use a 300-metre radius, a walking speed of
4.5 km/h, and no transfer penalty. The additional outputs are written to
`results/walking_sensitivity/` and do not replace the baseline outputs.

## Computational notes

The literature vulnerability measures require repeated node and edge removals
and can be considerably slower than the proposed measures. Set:

```python
RUN_LITERATURE_MEASURES = False
```

when only the proposed measures are required.

Map generation retrieves an OpenStreetMap background and therefore requires
internet access. It can be disabled with:

```python
RUN_MAPS = False
```

## Repository contents

- `reproduce_paper_results.py`: complete analysis and result-export workflow;
- `requirements.txt`: Python dependencies;
- `.gitignore`: files and generated outputs excluded from version control.

## License

No license has been selected yet. Add a `LICENSE` file before public release to
state how others may reuse the code.
