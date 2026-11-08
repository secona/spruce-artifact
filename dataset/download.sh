#!/usr/bin/env bash
# Run from the repository root. Existing archives are skipped.

wget -nc -P dataset https://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk01.tar.gz
wget -nc -P dataset https://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk17.tar.gz
wget -nc -P dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_4-4-4.tns.gz
wget -nc -P dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
