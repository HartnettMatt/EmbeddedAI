# WRITTEN BY CHATGPT 5-THINKING
# Prompt:
# Write a simple python script that iterates through a directory, finds all files of type .wav,
# then converts those .wav files from a 44.1kHz sample rate to a 16kHz sample rate without losing
# any information.


#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fractions import Fraction


def resample_to_16k(data: np.ndarray, sr_in: int) -> np.ndarray:
    """
    Resample audio to 16 kHz using a high-quality polyphase filter.
    Works for mono or multi-channel arrays shaped (samples,) or (samples, channels).
    """
    target_sr = 16000
    if sr_in == target_sr:
        return data

    # Find rational approximation for the resampling ratio
    ratio = Fraction(target_sr, sr_in).limit_denominator()
    up, down = ratio.numerator, ratio.denominator

    # Ensure 2D array with shape (samples, channels) for consistent axis handling
    if data.ndim == 1:
        data = data[:, None]

    # Polyphase resampling along time axis (axis=0) with built-in anti-aliasing
    resampled = resample_poly(data, up, down, axis=0)

    # Return to original dimensions if mono
    return resampled[:, 0] if resampled.shape[1] == 1 else resampled


def process_file(
    in_path: Path, out_path: Path, overwrite: bool = False, keep_subtype: bool = True
):
    if out_path.exists() and not overwrite:
        print(f"Skipping (exists): {out_path}")
        return

    # Read audio
    data, sr = sf.read(in_path, always_2d=False)
    info = sf.info(in_path)

    if sr == 16000:
        if overwrite and in_path.resolve() == out_path.resolve():
            print(f"Already 16 kHz, leaving as-is: {in_path}")
        else:
            # Copy without change
            sf.write(out_path, data, sr, subtype=info.subtype if keep_subtype else None)
            print(f"Copied (already 16 kHz): {out_path}")
        return

    # Convert to float64 for processing (soundfile returns float by default)
    data = np.asarray(data, dtype=np.float64)

    # Resample
    resampled = resample_to_16k(data, sr)

    # Write output (preserve original subtype if possible)
    sf.write(out_path, resampled, 16000, subtype=info.subtype if keep_subtype else None)
    print(f"Converted: {in_path.name} -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Find .wav files in a directory and convert from 44.1 kHz (or any rate) to 16 kHz."
    )
    parser.add_argument("directory", type=Path, help="Root directory to scan")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (mirrors input tree). Default: write next to each file.",
    )
    parser.add_argument(
        "--suffix",
        default="_16k",
        help="Suffix for output filenames (before .wav). Default: _16k",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output files if they exist"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write back to the same path (dangerous; implies --overwrite and ignores --suffix)",
    )
    args = parser.parse_args()

    root = args.directory
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    wavs = list(root.rglob("*.wav"))
    if not wavs:
        print("No .wav files found.")
        return

    for in_path in wavs:
        if args.inplace:
            out_path = in_path
            overwrite = True
        else:
            if args.outdir:
                # Mirror relative path under outdir
                rel = in_path.relative_to(root)
                out_dir = args.outdir / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = in_path.stem + args.suffix + ".wav"
                out_path = out_dir / out_name
            else:
                out_path = in_path.with_name(in_path.stem + args.suffix + ".wav")
            overwrite = args.overwrite

        process_file(in_path, out_path, overwrite=overwrite)


if __name__ == "__main__":
    main()
