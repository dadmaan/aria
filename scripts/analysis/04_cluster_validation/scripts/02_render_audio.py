#!/usr/bin/env python3
"""
MIDI to Audio Renderer - GHSOM Listening Test Audio Generation

This script converts MIDI files to high-quality audio using FluidSynth for
listening tests and perceptual evaluation.

Author: GHSOM Analysis Pipeline
Date: 2025-11-19
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


class MIDIRenderer:
    """Convert MIDI files to audio using FluidSynth."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        soundfont: Optional[Path] = None,
        sample_rate: int = 44100,
        format: str = "wav",
        normalize: bool = True,
        parallel_jobs: int = 4,
    ):
        """
        Initialize the MIDI renderer.

        Args:
            input_dir: Directory containing MIDI files
            output_dir: Output directory for audio files
            soundfont: Path to soundfont file (.sf2)
            sample_rate: Audio sample rate in Hz
            format: Output format (wav, flac, ogg)
            normalize: Apply audio normalization
            parallel_jobs: Number of parallel rendering processes
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.format = format
        self.normalize = normalize
        self.parallel_jobs = parallel_jobs

        # Find soundfont
        if soundfont and Path(soundfont).exists():
            self.soundfont = Path(soundfont)
        else:
            self.soundfont = self._find_soundfont()

        # Verify FluidSynth installation
        self._check_fluidsynth()

    def _find_soundfont(self) -> Optional[Path]:
        """Find available soundfont file."""
        common_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/share/soundfonts/default.sf2",
            "FluidR3_GM.sf2",
            "GeneralUser_GS.sf2",
            "/workspace/misc/GeneralUser-GS.sf2",  # https://github.com/ad-si/awesome-soundfonts?tab=readme-ov-file#soundfonts
        ]

        for path in common_paths:
            if Path(path).exists():
                print(f"Found soundfont: {path}")
                return Path(path)

        print("⚠ Warning: No soundfont found. Please specify with --soundfont")
        return None

    def _check_fluidsynth(self) -> bool:
        """Verify FluidSynth is installed and accessible."""
        try:
            result = subprocess.run(
                ["fluidsynth", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.split("\n")[0]
                print(f"FluidSynth found: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        print("✗ Error: FluidSynth not found. Please install:")
        print("  Ubuntu/Debian: sudo apt-get install fluidsynth fluid-soundfont-gm")
        print("  macOS: brew install fluid-synth")
        sys.exit(1)

    def find_midi_files(self) -> List[Tuple[Path, Path]]:
        """
        Find all MIDI files in input directory.

        Returns:
            List of (source_path, destination_path) tuples
        """
        midi_files = []

        # Search for MIDI files recursively
        for midi_file in self.input_dir.rglob("*.mid"):
            # Preserve directory structure
            rel_path = midi_file.relative_to(self.input_dir)
            output_file = self.output_dir / rel_path.with_suffix(f".{self.format}")
            midi_files.append((midi_file, output_file))

        # Also check for .midi extension
        for midi_file in self.input_dir.rglob("*.midi"):
            rel_path = midi_file.relative_to(self.input_dir)
            output_file = self.output_dir / rel_path.with_suffix(f".{self.format}")
            midi_files.append((midi_file, output_file))

        return sorted(midi_files)

    def render_midi(self, midi_file: Path, output_file: Path) -> Dict:
        """
        Render a single MIDI file to audio.

        Args:
            midi_file: Source MIDI file
            output_file: Destination audio file

        Returns:
            Status dictionary with success/error information
        """
        result = {
            "midi_file": str(midi_file),
            "output_file": str(output_file),
            "success": False,
            "error": None,
            "duration": None,
        }

        try:
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Build FluidSynth command
            cmd = [
                "fluidsynth",
                "-ni",  # Non-interactive mode
                "-g",
                "1.0",  # Gain
                "-r",
                str(self.sample_rate),  # Sample rate
                "-F",
                str(output_file),  # Output file
            ]

            if self.soundfont:
                cmd.extend([str(self.soundfont), str(midi_file)])
            else:
                cmd.append(str(midi_file))

            # Execute rendering
            render_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if render_result.returncode == 0 and output_file.exists():
                result["success"] = True
                result["file_size"] = output_file.stat().st_size

                # Get duration using ffprobe if available
                try:
                    duration_result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            str(output_file),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if duration_result.returncode == 0:
                        result["duration"] = float(duration_result.stdout.strip())
                except:
                    pass
            else:
                result["error"] = render_result.stderr or "Unknown error"

        except subprocess.TimeoutExpired:
            result["error"] = "Rendering timeout (>5 minutes)"
        except Exception as e:
            result["error"] = str(e)

        return result

    def render_batch(self) -> Dict:
        """
        Render all MIDI files in parallel.

        Returns:
            Rendering statistics and results
        """
        midi_files = self.find_midi_files()

        if not midi_files:
            print(f"✗ No MIDI files found in {self.input_dir}")
            return {"total_files": 0, "success": 0, "failed": 0, "results": []}

        print(f"\n{'='*60}")
        print(f"Found {len(midi_files)} MIDI files to render")
        print(f"Output format: {self.format}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Parallel jobs: {self.parallel_jobs}")
        print(f"{'='*60}\n")

        stats = {
            "total_files": len(midi_files),
            "success": 0,
            "failed": 0,
            "total_duration": 0.0,
            "total_size": 0,
            "start_time": datetime.now().isoformat(),
            "results": [],
        }

        # Render in parallel
        with ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
            futures = {
                executor.submit(self.render_midi, src, dst): (src, dst)
                for src, dst in midi_files
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                stats["results"].append(result)

                if result["success"]:
                    stats["success"] += 1
                    if result.get("duration"):
                        stats["total_duration"] += result["duration"]
                    if result.get("file_size"):
                        stats["total_size"] += result["file_size"]
                    status = "✓"
                else:
                    stats["failed"] += 1
                    status = "✗"

                # Progress update
                if completed % 10 == 0 or completed == len(midi_files):
                    print(
                        f"  [{completed}/{len(midi_files)}] {status} {Path(result['midi_file']).name}"
                    )

        stats["end_time"] = datetime.now().isoformat()

        print(f"\n{'='*60}")
        print("RENDERING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files: {stats['total_files']}")
        print(f"Successfully rendered: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        if stats["total_duration"] > 0:
            print(f"Total audio duration: {stats['total_duration']:.1f} seconds")
        if stats["total_size"] > 0:
            size_mb = stats["total_size"] / (1024 * 1024)
            print(f"Total size: {size_mb:.1f} MB")
        print(f"{'='*60}\n")

        return stats

    def save_manifest(self, stats: Dict) -> None:
        """Save rendering manifest with statistics."""
        manifest = {
            "rendering_config": {
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "soundfont": str(self.soundfont) if self.soundfont else None,
                "sample_rate": self.sample_rate,
                "format": self.format,
                "normalize": self.normalize,
                "parallel_jobs": self.parallel_jobs,
            },
            "statistics": {
                "total_files": stats["total_files"],
                "success": stats["success"],
                "failed": stats["failed"],
                "total_duration_seconds": stats["total_duration"],
                "total_size_bytes": stats["total_size"],
                "start_time": stats["start_time"],
                "end_time": stats["end_time"],
            },
            "failed_files": [r for r in stats["results"] if not r["success"]],
        }

        manifest_file = self.output_dir / "rendering_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved rendering manifest to {manifest_file}")

        # Save detailed results
        results_file = self.output_dir / "rendering_results.json"
        with open(results_file, "w") as f:
            json.dump(stats["results"], f, indent=2)
        print(f"Saved detailed results to {results_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Render MIDI files to audio for GHSOM listening tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for audio files",
    )
    parser.add_argument("--soundfont", type=Path, help="Path to soundfont file (.sf2)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        choices=[22050, 44100, 48000, 96000],
        help="Audio sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--format",
        choices=["wav", "flac", "ogg"],
        default="wav",
        help="Output audio format (default: wav)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable audio normalization"
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="Number of parallel rendering processes (default: 4)",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("MIDI TO AUDIO RENDERING - GHSOM Listening Test")
    print(f"{'='*70}\n")

    # Initialize renderer
    renderer = MIDIRenderer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        soundfont=args.soundfont,
        sample_rate=args.sample_rate,
        format=args.format,
        normalize=not args.no_normalize,
        parallel_jobs=args.parallel_jobs,
    )

    # Execute rendering
    stats = renderer.render_batch()
    renderer.save_manifest(stats)

    print(f"\n{'='*70}")
    print("✓ RENDERING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output directory: {args.output_dir}")
    print(f"Rendered {stats['success']} / {stats['total_files']} files\n")

    if stats["failed"] > 0:
        print(f"⚠ Warning: {stats['failed']} files failed to render")
        print(f"See {args.output_dir / 'rendering_manifest.json'} for details\n")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
