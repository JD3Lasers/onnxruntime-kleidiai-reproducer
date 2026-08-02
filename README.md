# ONNX Runtime Apple Silicon KleidiAI reproducer

This repository reproduces nondeterministic CPU Execution Provider output in
ONNX Runtime 1.27.0 on Apple Silicon. The same all-zero `float32` input is
submitted three times through one unchanged session. The third output differs
when KleidiAI is enabled and remains bit-identical when the documented
`mlas.disable_kleidiai=1` session option is applied.

## Environment

- macOS 26.5.2 (25F84), arm64, Apple M5 Pro
- Python 3.14.6
- NumPy 2.4.6
- ONNX Runtime CPU packages 1.27.0 and 1.26.0

## Run

Download `beatthis-final0.onnx` from the v1.0.0 release, then run:

```bash
python -m venv .venv
.venv/bin/pip install numpy==2.4.6 onnxruntime==1.27.0
.venv/bin/python reproduce_onnxruntime_kleidiai.py beatthis-final0.onnx
```

The model SHA-256 must be:

```text
e0024b9af8f9a0da5e2541e89ec92cfca82530aaaa33a9692d6f22626f0ebaf7
```

## Results

| ONNX Runtime | Session | All three equal | Changed output elements | Maximum absolute difference |
| --- | --- | --- | ---: | ---: |
| 1.27.0 | default | no | 1,500 | 0.0026569366455078125 |
| 1.27.0 | `mlas.disable_kleidiai=1` | yes | 0 | 0 |
| 1.26.0 | default | yes | 0 | 0 |
| 1.26.0 | `mlas.disable_kleidiai=1` | yes | 0 | 0 |

The 1.27.0 default output digests were:

```text
e7298cadec17c78236f1dd667f8010b8425949ebaa4a22e007fbe8f336ef9f0a
e7298cadec17c78236f1dd667f8010b8425949ebaa4a22e007fbe8f336ef9f0a
b889acda3c5687c629238f03f69cf0bfe7cc7c760fdbd9c64ad23f0a2e436474
```

The disabled control was stable at:

```text
1e2847f203fcb6b5e53c7973b929e387c4ca89d2ad7262736a7be550b1377a4c
```

## Model provenance

The attached ONNX file was converted from CPJKU/BeatThis `final0.ckpt` at
source commit `ad7974846029835307ba19a3d5cefbf40b243041`. BeatThis code and
published weights are MIT licensed. Conversion metadata is retained in
`beatthis_onnx_model_manifest.json`.

