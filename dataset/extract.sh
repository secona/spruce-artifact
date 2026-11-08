#!/usr/bin/env bash
# Run from the repository root. All downloaded archives are retained.
set -e

tar -xzf dataset/bcsstk01.tar.gz -C dataset --strip-components=1 bcsstk01/bcsstk01.mtx
tar -xzf dataset/bcsstk17.tar.gz -C dataset --strip-components=1 bcsstk17/bcsstk17.mtx
gzip -dkf dataset/matmul_4-4-4.tns.gz
gzip -dkf dataset/nell-2.tns.gz
