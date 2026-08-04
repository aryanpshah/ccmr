# Data: HVSMR-2.0

This project uses the **HVSMR-2.0** dataset (whole-heart CMR segmentation in congenital
heart disease). The dataset is **not** redistributed in this repository - download it
directly from the official figshare release below.

## Download

- **Collection:** https://doi.org/10.6084/m9.figshare.c.7074755
- **HVSMR-2.0 (cropped_norm)** - used by this pipeline: https://doi.org/10.6084/m9.figshare.25226363
  (files: `cropped_norm.zip`, `hvsmr_clinical.csv`, `hvsmr_technical.csv`)
- **HVSMR-2.0 (orig):** https://doi.org/10.6084/m9.figshare.25226360

The preprocessing scripts expect the `cropped_norm` NIfTI volumes and the clinical CSV
(`hvsmr_clinical.csv`). Place them under `data/raw/HVSMR2/` as described in the README.

## License

HVSMR-2.0 is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/

Under CC BY 4.0 you may share and adapt the data (including commercially), provided you
give appropriate credit, provide a link to the license, and indicate if changes were made.

## Required citation

> Pace, Danielle; Contreras, Hannah; Romanowicz, Jennifer; Ghelani, Shruti; Rahaman, Imon;
> Zhang, Yue; Gao, Patricia; Jubair, Mohammad Imrul; Yeh, Tom; Golland, Polina; Geva, Tal;
> Ghelani, Sunil; Powell, Andrew; Moghari, Mehdi Hedjazi (2024). *HVSMR-2.0: A 3D
> cardiovascular MR dataset for whole-heart segmentation in congenital heart disease.*
> figshare. Collection. https://doi.org/10.6084/m9.figshare.c.7074755

Associated paper: Pace et al., *Scientific Data* 11, 721 (2024).
https://doi.org/10.1038/s41597-024-03469-9

## Changes made to the data (CC BY "indicate changes")

The scripts in this repository transform the original HVSMR-2.0 data. Derived files are
generated locally by the pipeline and are **not** part of the original dataset:

- Resampling to 1 mm isotropic spacing; center crop/pad to a fixed size.
- Intensity normalization.
- Conversion to nnU-Net v2 dataset format and preprocessed tensors (`.b2nd`, `.pkl`).
- Generation of fixed train/val/test splits and nested label budgets (L5/L10/L20/L40).
