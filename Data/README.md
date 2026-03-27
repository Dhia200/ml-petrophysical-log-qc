# Data Directory

## Dataset Information

**Source:** Equinor Volve Field Open Data  
**Well:** 15/9-19  
**Location:** North Sea, Norway  
**License:** Creative Commons (CC BY-NC-SA 4.0)

## Download Link

The well log data is loaded directly from:
```
https://raw.githubusercontent.com/andymcdgeo/Petrophysics-Python-Series/master/Data/15-9-19_SR_COMP.LAS
```

## Data Attribution

This project uses publicly available data from the Equinor Volve Field dataset. All data remains property of Equinor ASA and is used in accordance with the Creative Commons license terms.

**Official Source:** https://www.equinor.com/energy/volve-data-sharing

## File Format

- **Format:** LAS (Log ASCII Standard)
- **Logs Available:** GR, CALI, DEN, NEU, AC, RDEP
- **Depth Range:** ~3000-4000 meters

## Usage

The Python script automatically downloads and processes the data. No manual download required.

```python
import lasio
las_url = "https://raw.githubusercontent.com/andymcdgeo/Petrophysics-Python-Series/master/Data/15-9-19_SR_COMP.LAS"
las = lasio.read(las_url)
```
