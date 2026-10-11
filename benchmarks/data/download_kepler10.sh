#!/usr/bin/env bash
# Download all long-cadence light curves of Kepler-10 (KIC 11904151) from MAST
set -e
cd "$(dirname "$0")" && mkdir -p kepler && cd kepler
U=https://archive.stsci.edu/missions/kepler/lightcurves/0119/011904151/
for f in $(curl -s $U | grep -o 'kplr[0-9-]*_llc\.fits' | sort -u); do curl -s -O "$U$f"; done
ls | wc -l
