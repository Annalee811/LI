#!/bin/sh
# Only disposable container filesystem paths; do not remove any /work data.
set -eu
rm -rf /usr/lib/gcc /usr/include /usr/share/doc /usr/share/man /usr/share/locale /usr/local/lib/python3.5/dist-packages/tensorflow/include /usr/local/lib/python3.5/dist-packages/notebook
find /usr/local/lib/python3.5 /usr/lib/python3.5 -name '*.pyc' -type f -delete
# Docker's vfs driver duplicates the image root. Relocate Python libraries into
# a bounded memory filesystem to leave disk space for real weights/predictions.
# Requires: --tmpfs /ram-python:rw,exec,size=1300m (exec permits shared-library mmap)
test -d /ram-python
mkdir -p /ram-python/usr/lib /ram-python/usr/local/lib
cp -a /usr/lib/python3.5 /ram-python/usr/lib/
cp -a /usr/local/lib/python3.5 /ram-python/usr/local/lib/
rm -rf /usr/lib/python3.5 /usr/local/lib/python3.5
export PYTHONHOME=/ram-python/usr
export PYTHONPATH=/ram-python/usr/local/lib/python3.5/dist-packages
exec /usr/bin/python3.5 analysis-scripts/0353-2-B3_segmentation_pilot/binary_continuation_v25/train_tf.py "$@"
