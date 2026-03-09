# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:59:16 2025

@author: sahme627
"""

#!/usr/bin/env python3
"""
onnx_infer.py
Run DTOF inference with ONNX Runtime.

Usage:
  python onnx_infer.py -i input.mat -o pred.mat -m resendc.onnx --rows 51
"""

import argparse
import numpy as np
from scipy.io import loadmat, savemat
import onnxruntime as ort
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', required=True, help='Input .mat (must contain variable "tmp")')
    ap.add_argument('-o','--output', required=True, help='Output .mat path')
    ap.add_argument('-m','--model', required=True, help='ONNX model path')
    ap.add_argument('--rows', type=int, default=51, help='Use tmp[1:rows, :] (default 51)')
    args = ap.parse_args()

    inp_path = Path(args.input)
    onnx_path = Path(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load .mat and preprocess like your pipeline
    mat = loadmat(str(inp_path))
    if 'tmp' not in mat:
        raise RuntimeError(f"Variable 'tmp' not found in {inp_path}")
    tmp = mat['tmp']                       # shape: (R_all, T)
    R = min(args.rows, tmp.shape[0])
    x = tmp[:R, :].T.astype(np.float32)    # (T, R)  == (T, 51) typically
    X = x[None, ...]                       # (1, T, R)

    # ONNX Runtime session (CPU by default)
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Run inference: expect output (1, T, 256)
    y = sess.run([out_name], {in_name: X})[0]
    if y.ndim == 3:
        y = np.squeeze(y, axis=0)          # (T, 256)
    # Save as (256, T) to match your convention
    prediction = y.T                       # (256, T)

    savemat(str(out_path), {'prediction': prediction})
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
