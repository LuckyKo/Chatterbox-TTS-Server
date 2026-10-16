#!/usr/bin/env python3
# File: chatterbox_tts.py
# One-shot Chatterbox TTS CLI for AgentCascade media_gen integration.
#
# This is a "lite" quick-start entry point that mirrors the koboldcpp one-shot
# binary pattern: load the model, synthesize exactly once, write the encoded
# audio bytes to STDOUT, then exit. Because the process exits, the OS reclaims
# all VRAM/RAM between AgentCascade LLM runs -- nothing stays resident.
#
# CONTRACT (what AgentCascade's media_gen relies on):
#   * STDOUT carries ONLY the raw encoded audio bytes (mp3/wav/opus). No logs,
#     no banners, no progress bars.
#   * ALL diagnostics (logging, errors, stray prints) go to STDERR.
#   * Exit code 0 on success; non-zero on any failure (see EXIT CODES below).
#
# Usage (argv shape media_gen should build):
#   python chatterbox_tts.py --text "..." --reference /abs/path/voice.wav \
#       [--device cpu|cuda|mps|auto] [--output-format mp3|wav|opus] \
#       [--temperature F] [--exaggeration F] [--cfg-weight F] [--seed INT] \
#       [--target-sample-rate INT]
#   Long text may also be piped via stdin when --text is omitted.
#
# EXIT CODES:
#   0  success (audio bytes on stdout)
#   1  unexpected error during load/synth/encode
#   2  bad arguments / missing reference file / no input text
#   3  model load failure
#   4  synthesis failure
#   5  audio encoding failure
#   6  dependency/import failure (chatterbox, pydub, soundfile, ...)
#
# RUNTIME DEPENDENCIES:
#   * The `chatterbox` model package (imported by engine.py) and the cached
#     model under models/ (HF cache layout).
#   * For --output-format mp3: encode_audio() uses pydub.AudioSegment.export(
#     format="mp3"), which shells out to an external FFmpeg binary. FFmpeg must
#     be present on PATH for mp3 output. wav/opus use soundfile/libsndfile and
#     do NOT require FFmpeg.

import argparse
import logging
import sys

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Heavy, repo-local imports. Guarded so a missing dependency yields a clean
# stderr diagnostic (exit 6) instead of a traceback -- stdout must stay clean.
# ---------------------------------------------------------------------------
# engine.py emits bare print() at import time on its ImportError path (engine.py:20-22),
# which would otherwise reach real stdout. Route stdout to stderr across the import
# block so the exit-6 path stays clean, then restore real stdout.
_real_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    import engine  # load_model(), synthesize()
    import utils  # encode_audio()
    from config import (
        config_manager,
        _set_nested_value,
        get_audio_sample_rate,
    )
except Exception as _import_exc:  # ImportError or import-time side-effect failure
    print(f"chatterbox_tts: failed to import engine/utils/config: {_import_exc}", file=sys.stderr)
    sys.exit(6)
finally:
    sys.stdout = _real_stdout


def resolve_text(args: argparse.Namespace) -> str | None:
    """Return the text to synthesize: --text, else piped stdin, else None."""
    if args.text is not None:
        return args.text
    # Only fall back to stdin when it is a real pipe (not an interactive TTY).
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return None


def apply_device_override(device: str) -> None:
    """Force the TTS device into the in-memory config so engine.load_model()
    picks it up. load_model() reads config_manager.get_string(
    "tts_engine.device", "auto"), so writing the nested key is sufficient and
    requires no file write or monkeypatch. The repo's own _set_nested_value
    helper is reused to stay consistent with config.py conventions."""
    _set_nested_value(config_manager.config, ["tts_engine", "device"], device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot Chatterbox TTS: write audio bytes to stdout and exit.",
    )
    parser.add_argument("--text", help="Text to synthesize (or pipe via stdin).")
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference/voice audio file (passed as audio_prompt_path).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Override the config device. If omitted, the config value is used.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-format",
        choices=["mp3", "wav", "opus"],
        default="mp3",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        help="Resample target rate. If omitted, uses config audio_output.sample_rate "
        "or the model's native rate (no resample).",
    )
    args = parser.parse_args()

    # Keep stdout pristine: logging defaults to a stderr sink, and we raise the
    # root level to WARNING so only WARNING/ERROR/CRITICAL surface (on stderr).
    logging.getLogger().setLevel(logging.WARNING)

    # --- Validate inputs (cheap, before any heavy work) ---
    text = resolve_text(args)
    if not text:
        print("chatterbox_tts: no input text (--text empty/omitted and stdin empty/TTY).", file=sys.stderr)
        return 2

    ref_path = Path(args.reference)
    if not ref_path.is_file():
        print(f"chatterbox_tts: reference file not found: {ref_path}", file=sys.stderr)
        return 2

    # --- Heavy phase: load / synthesize / encode ---
    # Any stray print() from libraries during this phase must NOT reach stdout,
    # so temporarily route sys.stdout to stderr. The real stdout is restored
    # before we write the audio payload.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if args.device:
            apply_device_override(args.device)

        if not engine.load_model():
            print("chatterbox_tts: model load failed.", file=sys.stderr)
            return 3

        wav_tensor, sr = engine.synthesize(
            text=text,
            audio_prompt_path=str(ref_path),
            temperature=args.temperature,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
        )
        if wav_tensor is None or sr is None:
            print("chatterbox_tts: synthesis failed.", file=sys.stderr)
            return 4

        # sr is the model's native sample rate (chatterbox_model.sr) and MUST be
        # the first arg to encode_audio -- it describes the tensor we just got.
        # engine.synthesize already returns a CPU tensor, so .cpu() is redundant;
        # keep .detach(). Squeeze only when ndim>1 so a 1-D (N,) waveform is never
        # collapsed to a 0-d scalar.
        audio_array = np.asarray(wav_tensor.detach().numpy(), dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        # target_sample_rate: explicit flag wins; else config's configured rate;
        # else None (encode at native rate, no resample).
        target_sr = args.target_sample_rate
        if target_sr is None:
            cfg_sr = get_audio_sample_rate()
            if isinstance(cfg_sr, int) and cfg_sr > 0:
                target_sr = cfg_sr

        audio_bytes = utils.encode_audio(
            audio_array,
            sr,
            output_format=args.output_format,
            target_sample_rate=target_sr,
        )
        if audio_bytes is None:
            print("chatterbox_tts: audio encoding failed.", file=sys.stderr)
            return 5
    except Exception as _heavy_exc:
        print(f"chatterbox_tts: unexpected error during load/synth/encode: {_heavy_exc}", file=sys.stderr)
        return 1
    finally:
        sys.stdout = _real_stdout

    # --- Emit the payload: stdout carries ONLY the audio bytes. ---
    sys.stdout.buffer.write(audio_bytes)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
