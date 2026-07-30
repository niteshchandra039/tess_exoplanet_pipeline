# Chapter 5
# Light Curve Acquisition

---

# 5.1 Overview

Following target identification and literature ephemeris retrieval, the pipeline proceeds to acquire the photometric observations from which transit signals will be detected.

The objective of this stage is to obtain a scientifically calibrated collection of TESS light curves that are suitable for subsequent preprocessing and transit analysis.

Unlike many TESS pipelines that always download observations from MAST, the present implementation first attempts to reuse locally available FITS files before contacting the archive. This design substantially reduces download time, enables completely offline processing after the first download, and improves reproducibility.

The acquisition stage supports both

- online retrieval from MAST using Lightkurve, and
- direct loading of previously downloaded FITS files.

The output of this stage is always a

```python
lightkurve.LightCurveCollection
```

which becomes the input to the preprocessing stage.

---

# 5.2 Position within the Pipeline

The acquisition module occupies the following position within the overall workflow.

```mermaid
flowchart LR

A[Target Resolution]

-->

B[Archive Query]

-->

C[Light Curve Acquisition]

-->

D[Preprocessing]

-->

E[TLS/BLS Detection]

-->

F[Bayesian Inference]
```

At this point, the pipeline already knows

- the TIC identifier,
- the target coordinates,
- any published orbital periods,
- the available observation sectors.

Its remaining task is therefore to retrieve the corresponding TESS observations.

---

# 5.3 Software Architecture

The acquisition subsystem is implemented primarily in

```
tess_pipeline/data/download.py
```

and relies heavily upon the Lightkurve package for communication with MAST.

The overall architecture is

```text
download_lightcurves()

│

├──────────────► Search local FITS

│

├──────────────► Search MAST

│

├──────────────► Sector Selection

│

├──────────────► Download Missing Files

│

├──────────────► Read FITS

│

└──────────────► LightCurveCollection
```

Unlike preprocessing or transit detection, this stage performs almost no scientific processing.

Its responsibility is restricted to data acquisition and validation.

---

# 5.4 Design Philosophy

The acquisition module was designed around several principles.

## Offline capability

Previously downloaded observations should never be downloaded again unless explicitly requested.

---

## Reproducibility

The exact FITS files used during analysis remain available locally.

---

## Flexible sector selection

Users should be able to request

- one sector,
- several sectors,
- all sectors,
- contiguous sectors,
- arbitrary sector lists.

---

## Minimal assumptions

Only calibrated SPOC PDCSAP light curves are downloaded.

No detrending occurs during acquisition.

---

# 5.5 Primary Entry Point

All downloads begin inside

```python
download_lightcurves()
```

whose simplified signature is

```python
download_lightcurves(

tic_id,

author="SPOC",

cadence=120,

sectors=1,

force_download=False

)
```

The function performs four major operations

1. locate existing observations,
2. determine required sectors,
3. download missing observations,
4. return a LightCurveCollection.

---

# 5.6 High-Level Algorithm

The overall acquisition procedure can be summarized by the following flowchart.

```mermaid
flowchart TD

Start

-->

Search Local FITS

-->

Any Local Files?

Any Local Files?

-->|Yes|

Determine Available Sectors

Any Local Files?

-->|No|

Search MAST

Search MAST

-->

Determine Available Sectors

-->

Sector Selection

-->

Required Files Local?

Required Files Local?

-->|Yes|

Load FITS

Required Files Local?

-->|No|

Download Missing Files

Download Missing Files

-->

Load FITS

Load FITS

-->

LightCurveCollection

-->

Return
```

---

# 5.7 Local FITS Discovery

Before contacting MAST, the pipeline scans

```
data/fits/
```

for previously downloaded observations.

Two filename formats are recognised.

```
*TIC*

```

and

```
*zero-padded TIC*

```

where

```
307210830

↓

0000000307210830
```

This makes the loader compatible with multiple generations of SPOC filenames.

The search operation therefore resembles

```python
glob("*-307210830-*")

glob("*-0000000307210830-*")
```

Duplicate filenames are automatically removed.

---

# 5.8 Determining Sector Numbers

Each FITS file must be associated with its observing sector.

The pipeline attempts two independent methods.

## Method 1

Read the filename

```
-s0027-
```

↓

Sector 27

---

## Method 2

If the filename does not contain the sector,

the FITS header is inspected.

```
SECTOR

↓

27
```

This dual strategy ensures compatibility with observations originating from different processing pipelines.

The resulting mapping is

```text
Sector

↓

Files

↓

{

27 : [file1],

28 : [file2],

29 : [file3]

}
```

which is subsequently used during sector selection.

---

# 5.9 Online Archive Search

If internet access is available, the pipeline queries MAST using

```python
lightkurve.search_lightcurve()
```

with

```python
mission="TESS"

author="SPOC"

exptime=120
```

Only calibrated SPOC observations are requested.

The returned search table provides

- observing sectors,
- observation identifiers,
- download URLs,
- cadence information.

If MAST cannot be reached but suitable FITS files already exist locally, the pipeline continues entirely offline.

If neither local files nor online observations are available, execution terminates with

```python
NoCadenceDataError
```

---

# 5.10 Offline Operation

One particularly elegant aspect of the implementation is its graceful degradation.

```mermaid
flowchart LR

MAST Available?

MAST Available?

-->|Yes|

Online Mode

MAST Available?

-->|No|

Local FITS Available?

Local FITS Available?

-->|Yes|

Offline Processing

Local FITS Available?

-->|No|

Abort
```

This behaviour makes the pipeline suitable for execution on HPC clusters or remote observatories where internet access may be intermittent.

---

# End of Part 1