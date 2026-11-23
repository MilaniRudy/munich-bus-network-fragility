# Munich Bus Network Fragility Analysis

This script analyses the Munich bus network using GTFS data from MVG. It builds service-level and aggregated network models, computes fragility and criticality measures, and exports the results as tables and figures.

## Requirements

Python 3.10 or later is recommended. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

Adjust the parameters near the top of `backup_fragility_munich.py` if needed, particularly the analysis date, time window, output options, and sensitivity-analysis settings. Then run:

```bash
python backup_fragility_munich.py
```

The script downloads the current MVG GTFS archive automatically. A local GTFS ZIP file can also be used by changing `GTFS_SOURCE`.

## Output

Results are saved in the `results/` directory and include CSV tables, an Excel workbook, figures, rankings, path statistics, and optional runtime and walking-sensitivity reports.
