#!/bin/sh
# Download the two TESS light curves used by the regression numbers in wotan/tests.py
# (TESS SPOC data alerts, sector 1, from MAST). Run from anywhere:
#
#     sh tests/data/download_tess_data.sh
#
# The files (~2 MB each) are saved next to this script and are git-ignored.
set -eu
cd "$(dirname "$0")"
BASE="https://archive.stsci.edu/hlsps/tess-data-alerts"
for f in \
    hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits \
    hlsp_tess-data-alerts_tess_phot_00207081058-s01_tess_v1_lc.fits
do
    curl --fail --location --retry 3 --output "$f" "$BASE/$f"
done
ls -l *.fits
