# Chapter 3
# Target Resolution

---

# 3.1 Overview

The first stage of the pipeline is **Target Resolution**.

Its purpose is to convert user input into a standardized internal representation that can be used by every downstream module.

Unlike many pipelines that simply accept a TIC number, the current implementation is capable of resolving targets from

- TIC IDs
- local FITS files
- FITS directories
- FITS headers
- FITS filenames
- local archive metadata
- remote MAST metadata (optional)

The output of this stage is a canonical target dictionary

```python
{
    "tic_id": int,
    "name": str,
    "ra": float | None,
    "dec": float | None,
}
```

This dictionary becomes the foundation for the remainder of the pipeline.

---

# 3.2 Objectives

The target resolution stage has four scientific objectives.

1. Identify the target TIC ID.

2. Determine celestial coordinates.

3. Standardize target naming.

4. Gather sufficient metadata for downstream archive and light curve queries.

---

# 3.3 Inputs

The module accepts several forms of input.

| Input | Example |
|---------|----------------|
| Integer TIC | `307210830` |
| TIC string | `"TIC 307210830"` |
| Compact TIC string | `"TIC307210830"` |
| FITS file | `sector44.fits` |
| Directory | `./LightCurves/` |
| Multiple FITS files | `[file1,file2]` |

This flexibility greatly simplifies usage.

---

# 3.4 High-Level Workflow

The complete workflow is

```mermaid
flowchart TD

Start

-->

Input

-->

Decision{Input Type}

Decision -->|TIC| Parse TIC

Decision -->|FITS| Read FITS

Parse TIC --> Lookup Coordinates

Read FITS --> Read Header

Read Header --> TIC Found?

TIC Found? -->|Yes| Read Coordinates

TIC Found? -->|No| Filename Search

Filename Search --> TIC Found 2?

TIC Found 2? -->|Yes| Read Coordinates

TIC Found 2? -->|No| Fallback Target

Fallback Target --> Parse TIC

Read Coordinates --> Missing Coordinates?

Missing Coordinates? -->|Yes| Local Archive

Local Archive --> Still Missing?

Still Missing? -->|Yes| Remote MAST

Still Missing? -->|No| Finish

Remote MAST --> Finish
```

---

# 3.5 Internal Functions

The module is composed of several small helper functions.

```text
resolve_target()

resolve_target_from_fits()

expand_fits_paths()

infer_tic_id_from_fits_paths()

infer_coordinates_from_fits_paths()

_lookup_local_coordinates()

_fetch_coordinates()

get_observation_metadata()
```

Each function performs exactly one task.

This keeps the code highly modular.

---

# 3.6 FITS Path Expansion

The first utility converts arbitrary user input into a list of FITS files.

Input

```python
Path

directory

list

tuple
```

↓

Output

```python
list[Path]
```

Algorithm

```text
Directory?

↓

Yes

↓

glob("*.fits*")

↓

Collect Files

↓

Remove Duplicates
```

Duplicate removal is performed using

```python
dict.fromkeys(...)
```

which preserves ordering while eliminating repeated files.

---

# 3.7 TIC ID Extraction

The pipeline attempts to infer the TIC ID using several increasingly general methods.

## Step 1

Read FITS headers.

Header keywords searched

```text
TICID

TIC

OBJECT

TARGETID
```

The search stops immediately once a valid TIC is found.

---

## Step 2

If headers fail,

the filename is inspected.

Example

```text
tess2020440105921-s0030-000000307210830-0123.fits
```

↓

extract

```text
307210830
```

using regular expressions.

---

## Step 3

General TIC parser

```
(?:tic[\s_-]*)?(\d{6,16})
```

This accepts

```text
TIC307210830

tic 307210830

tic_307210830

307210830
```

making the parser highly tolerant.

---

# 3.8 Coordinate Resolution

Coordinates are obtained hierarchically.

```mermaid
flowchart TD

FITS Header

-->

Available?

Available? -->|Yes| Use FITS

Available? -->|No| Local Archive

Local Archive

-->

Available2?

Available2? -->|Yes| Use Archive

Available2? -->|No| Remote MAST

Remote MAST --> Finish
```

This minimizes unnecessary internet requests.

---

# 3.9 Reading FITS Coordinates

Several common FITS keywords are supported.

RA

```text
RA_OBJ

RA_TARG

RA

S_RA

OBJCTRA
```

Dec

```text
DEC_OBJ

DEC_TARG

DEC

S_DEC

OBJCTDEC
```

If

```text
OBJCTRA

OBJCTDEC
```

are encountered,

they are interpreted as

```python
Angle(..., unit="hourangle")
```

otherwise

they are assumed already in degrees.

This improves compatibility with heterogeneous FITS products.

---

# 3.10 Local Archive Lookup

If coordinates are absent,

the pipeline queries the locally stored NASA archive export.

```python
get_local_archive_record()
```

returns

```text
RA

Dec
```

without requiring internet access.

Advantages

- extremely fast

- reproducible

- works offline

---

# 3.11 Remote Coordinate Lookup

Only if the previous methods fail,

Lightkurve is used.

```python
search_lightcurve(
    "TIC XXXXX",
    mission="TESS"
)
```

Coordinates are extracted from

```python
table["s_ra"]

table["s_dec"]
```

This serves as the final fallback.

---

# 3.12 Observation Metadata

The module also provides

```python
get_observation_metadata()
```

which determines

```text
Available sectors

Number of sectors

Whether observations exist
```

using

```python
lightkurve.search_lightcurve()
```

Output

```python
{
    "sectors":[...],

    "has_data":True,

    "n_sectors":12
}
```

This information is later used during download.

---

# 3.13 Complete Data Flow

```mermaid
flowchart LR

User

-->

Target Input

-->

Parser

-->

TIC

-->

Coordinates

-->

Metadata

-->

PipelineResults.target
```

The resulting target dictionary becomes the input to every later stage.

---

# 3.14 Error Handling

The module throws

```python
TargetResolutionError
```

when

- TIC cannot be parsed

- FITS headers contain no usable identifiers

- no fallback target exists

This prevents later failures during archive queries.

---

# 3.15 Computational Complexity

Most operations are constant time.

| Operation | Complexity |
|------------|------------|
| Regex parsing | O(n) |
| FITS header reading | O(H) |
| Directory expansion | O(F) |
| Coordinate lookup | O(1) |
| Metadata query | Network bound |

where

- n = string length

- H = number of header cards

- F = number of FITS files

The dominant cost is network latency during remote MAST queries.

---

# 3.16 Strengths

The implementation has several notable strengths.

### Flexible Input

Supports both raw TIC IDs and local FITS products.

---

### Offline Capability

Coordinates can be obtained without internet access.

---

### Hierarchical Lookup

Information is obtained from the cheapest source first.

```text
Header

↓

Archive

↓

Remote
```

This minimizes latency.

---

### Good FITS Compatibility

Many common FITS conventions are recognized.

---

### Clean Modular Design

Each helper function performs exactly one task.

Testing is straightforward.

---

# 3.17 Current Limitations

Several limitations are apparent.

## Only TIC IDs are Supported

The parser accepts only TIC identifiers.

It cannot currently resolve

- Gaia IDs

- TOI numbers

- HD names

- HIP identifiers

- SIMBAD object names

- RA/Dec coordinate strings

---

## No Proper Motion Handling

Coordinates are assumed static.

High proper motion stars are not propagated to the TESS epoch.

---

## First Match Wins

When multiple FITS files are supplied,

the first valid TIC is immediately accepted.

Consistency between files is never verified.

---

## Sequential FITS Reading

Large directories are scanned serially.

Parallel FITS parsing could substantially reduce runtime.

---

## Remote Lookup Uses Search Results

Coordinates are extracted from Lightkurve search tables.

Direct TIC catalog queries through Astroquery MAST would be more robust and scientifically explicit.

---

# 3.18 Recommended Improvements

The following enhancements are recommended.

### Native Object Resolver

Support

```text
Gaia

TOI

HIP

HD

2MASS

SIMBAD
```

through a unified resolver.

---

### Coordinate Inputs

Accept

```text
RA Dec

SkyCoord

J2000 strings
```

directly.

---

### Proper Motion Propagation

Propagate Gaia coordinates to the TESS observation epoch.

---

### Parallel FITS Parsing

Use concurrent header reading for large datasets.

---

### Metadata Cache

Cache

```text
coordinates

observation metadata

archive queries
```

to avoid repeated network requests.

---

# 3.19 Summary

The Target Resolution stage provides a robust and flexible entry point to the pipeline. It standardizes heterogeneous user inputs into a canonical target representation while minimizing unnecessary remote queries through a hierarchical lookup strategy. The implementation is modular, efficient, and well suited for TIC-based analyses.

Its primary limitations stem from the restricted identifier support and lack of astrometric propagation rather than deficiencies in software architecture. Extending the resolver to support multiple astronomical catalogs and coordinate-based searches would significantly broaden the applicability of the pipeline without requiring major architectural changes.

The next chapter examines how the resolved target information is used to retrieve published planetary ephemerides from the NASA Exoplanet Archive and how these archival constraints influence the transit search.