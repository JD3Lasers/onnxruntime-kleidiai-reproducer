"""Reproduce ONNX Runtime Arm KleidiAI drift with identical generated input."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import numpy.typing as npt

_INPUT_NAME = "input_spectrogram"
_OUTPUT_NAME = "beat"
_INPUT_SHAPE = (1, 1500, 128)
_REPLAY_COUNT = 3
_KLEIDIAI_DISABLED_ENTRY = ("mlas.disable_kleidiai", "1")


class KleidiAIReplaySummary(TypedDict):
    output_sha256: list[str]
    all_equal: bool
    changed_output_elements: int
    max_absolute_difference: float


class KleidiAIReplayReport(TypedDict):
    onnxruntime_version: str
    platform: str
    machine: str
    model_file_name: str
    model_sha256: str
    input_shape: list[int]
    replay_count: int
    default_kleidiai: KleidiAIReplaySummary
    kleidiai_disabled: KleidiAIReplaySummary


class _SessionOptions(Protocol):
    intra_op_num_threads: int

    def add_session_config_entry(self, key: str, value: str) -> None: ...


class _InferenceSession(Protocol):
    def run(self, output_names: list[str], feed: dict[str, object]) -> list[object]: ...


class _OnnxRuntimeModule(Protocol):
    __version__: str

    def SessionOptions(self) -> _SessionOptions: ...

    def InferenceSession(
        self,
        model_path: str,
        *,
        sess_options: _SessionOptions,
        providers: list[str],
    ) -> _InferenceSession: ...


def build_kleidiai_replay_report(
    model_path: Path,
    *,
    onnxruntime_module: _OnnxRuntimeModule | None = None,
) -> KleidiAIReplayReport:
    """Return identical-input replay evidence with default and disabled KleidiAI."""
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    if onnxruntime_module is None:
        onnxruntime_module = cast(
            _OnnxRuntimeModule,
            importlib.import_module("onnxruntime"),
        )
    model_sha256 = _file_sha256(model_path)
    input_spectrogram = np.zeros(_INPUT_SHAPE, dtype=np.float32)
    return KleidiAIReplayReport(
        onnxruntime_version=onnxruntime_module.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        model_file_name=model_path.name,
        model_sha256=model_sha256,
        input_shape=list(_INPUT_SHAPE),
        replay_count=_REPLAY_COUNT,
        default_kleidiai=_replay_summary(
            onnxruntime_module,
            model_path,
            input_spectrogram,
            disable_kleidiai=False,
        ),
        kleidiai_disabled=_replay_summary(
            onnxruntime_module,
            model_path,
            input_spectrogram,
            disable_kleidiai=True,
        ),
    )


def _replay_summary(
    onnxruntime_module: _OnnxRuntimeModule,
    model_path: Path,
    input_spectrogram: npt.NDArray[np.float32],
    *,
    disable_kleidiai: bool,
) -> KleidiAIReplaySummary:
    options = onnxruntime_module.SessionOptions()
    options.intra_op_num_threads = 4
    if disable_kleidiai:
        options.add_session_config_entry(*_KLEIDIAI_DISABLED_ENTRY)
    session = onnxruntime_module.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    outputs = tuple(
        np.asarray(session.run([_OUTPUT_NAME], {_INPUT_NAME: input_spectrogram})[0])
        for _ in range(_REPLAY_COUNT)
    )
    first_output = outputs[0]
    output_sha256 = [hashlib.sha256(output.tobytes()).hexdigest() for output in outputs]
    first_output_bytes = np.frombuffer(
        first_output.tobytes(),
        dtype=np.uint8,
    ).reshape(first_output.size, first_output.dtype.itemsize)
    changed_output_elements = np.zeros(first_output.size, dtype=np.bool_)
    max_absolute_difference = 0.0
    for output in outputs[1:]:
        output_bytes = np.frombuffer(output.tobytes(), dtype=np.uint8).reshape(
            output.size,
            output.dtype.itemsize,
        )
        changed_output_elements |= np.any(first_output_bytes != output_bytes, axis=1)
        absolute_difference = np.abs(first_output - output)
        max_absolute_difference = max(
            max_absolute_difference,
            float(np.max(absolute_difference)),
        )
    return KleidiAIReplaySummary(
        output_sha256=output_sha256,
        all_equal=len(set(output_sha256)) == 1,
        changed_output_elements=int(np.count_nonzero(changed_output_elements)),
        max_absolute_difference=max_absolute_difference,
    )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as model_file:
        return hashlib.file_digest(model_file, "sha256").hexdigest()


def main(
    argv: Sequence[str] | None = None,
    *,
    onnxruntime_module: _OnnxRuntimeModule | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    namespace = parser.parse_args(argv)
    report = build_kleidiai_replay_report(
        namespace.model_path,
        onnxruntime_module=onnxruntime_module,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
