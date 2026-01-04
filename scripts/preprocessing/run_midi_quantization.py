import argparse
import random

import joblib
from src.preprocessing.quantization import *
from utility.os_utils import add_suffix_to_basename, read_files_from_directory

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert audio files to MIDI using basic-pitch."
)

# Add command-line arguments
parser.add_argument(
    "--data_dir", required=True, help="Path to the input data directory."
)
parser.add_argument(
    "--output_dir",
    required=False,
    default=None,
    help="Path to the output directory for MIDI files.",
)
parser.add_argument(
    "--extensions",
    nargs="+",
    default=["mid"],
    help="List of file extensions to process.",
)
parser.add_argument(
    "--num_cores",
    type=int,
    default=-1,
    help="Number of CPU cores to use for parallel processing.",
)
parser.add_argument(
    "--sample_size", type=int, help="Select smaller number of samples to process."
)
parser.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="Select smaller number of samples to process.",
)
parser.add_argument(
    "--beat_resolution",
    type=int,
    default=480,
    help="The number of ticks per beat in the generated quantized MIDI file.",
)
parser.add_argument(
    "--step_size",
    type=float,
    default=0.25,
    help="The quantization factor in multiples or fractions of quarter notes.",
)
parser.add_argument(
    "--velocity",
    type=int,
    default=80,
    help="The velocity (volume) of the notes in the quantized MIDI file.",
)
parser.add_argument(
    "--override_timing",
    type=bool,
    default=False,
    help="Override timing and tempo of the track to 120bpm and 4/4.",
)


# Parse the command-line arguments
args = parser.parse_args()


def check_tempo_range(
    tempo_microseconds, min_tempo_microseconds=1500000, max_tempo_microseconds=250000
):
    """
    Check a tempo value in microseconds per quarter note and set to a defined range.

    Parameters:
    - tempo_microseconds (int): The original tempo value in microseconds per quarter note.
    - min_tempo_microseconds (int, optional): The minimum tempo in microseconds per quarter note. Defaults to 1,500,000 equals to 40 BPM.
    - max_tempo_microseconds (int, optional): The maximum tempo in microseconds per quarter note. Defaults to 250,000 equals to 240 BPM.
    - threshold_microseconds (int, optional): The threshold for considering a tempo as too high or too low. Defaults to 1,000,000.

    Returns:
    int: The normalized tempo value in microseconds per quarter note within the specified range.

    Notes:
    - This function maps the original tempo value to the standard range defined by min_tempo_microseconds and max_tempo_microseconds.
    - If the original tempo is below the defined range, it is set to the minimum tempo.
    - If the original tempo is above the defined range, it is set to the maximum tempo.
    """
    # Ensure the tempo is within the specified range
    if tempo_microseconds > min_tempo_microseconds:
        return min_tempo_microseconds
    elif tempo_microseconds < max_tempo_microseconds:
        return max_tempo_microseconds
    else:
        return tempo_microseconds


def get_tempo(mid):
    """
    Get the tempo and time signature information from a MIDI file.

    Parameters:
    - mid (mido.MidiFile): A Mido MidiFile object representing a MIDI file.

    Returns:
    int or None: The tempo of the MIDI file in microseconds per quarter note.
        Returns None if no 'set_tempo' message is found.


    Notes:
    - This function iterates through the events in the first track of the MIDI file
    to find the first occurrence of 'set_tempo' message.
    - The 'set_tempo' message contains the tempo information in microseconds per quarter note.
    - If a 'set_tempo' message is found, the tempo value is returned.
    - If no 'set_tempo' message is found, returns None.
    """
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            return check_tempo_range(msg.tempo)
    return None


def quantize_midi_file(
    file_path,
    output_dir,
    verbose=args.verbose,
    beat_resolution=args.beat_resolution,
    step_size=args.step_size,
    velocity=args.velocity,
):
    """
    Quantize a MIDI file to fit a desired grid.

    Parameters:
    - file_path (str): Path to the MIDI file to be quantized.
    - output_dir (str): Directory to save the final file.
    - verbose (bool, optional): If True, print additional information during the quantization process.
                               Defaults to True.
    - beat_resolution (int, optional): The number of ticks per beat in the generated quantized MIDI file.
                                      Defaults to 480, a common value in many MIDI files.
    - step_size (float, optional): The quantization factor in multiples or fractions of quarter notes.
                                  Defaults to 0.25.
    - velocity (int, optional): The velocity (volume) of the notes in the quantized MIDI file.
                               Defaults to 80.
    - override_time_info (bool, optional): Override original tempo and time signature.

    Returns:
    Mido.MidiFile: The quantized and reformatted MIDI object.

    Notes:
    - This function quantizes a MIDI file to fit the desired grid.
    - The `file_path` parameter specifies the path to the MIDI file to be quantized.
    - The `verbose` parameter controls whether additional information is printed during the quantization process.
    - The `beat_resolution` parameter determines the timing resolution of the quantized MIDI file.
    - The `step_size` parameter determines the granularity of the quantization grid.
    - The `velocity` parameter sets the default velocity for note events in the quantized MIDI file.
    - The quantization process involves reformatting the MIDI file, converting it to a matrix,
      quantizing the matrix, converting it back to a quantized MIDI object, and reformatting the result.
    - The final quantized and reformatted MIDI object is returned.
    """
    input_midi_path = file_path

    if output_dir:
        output_midi_path = file_path.replace(args.data_dir, args.output_dir)
        os.makedirs(os.path.split(output_midi_path)[0], exist_ok=True)
        output_midi_path = add_suffix_to_basename(output_midi_path, suffix="_quantized")

    else:
        output_midi_path = add_suffix_to_basename(input_midi_path, suffix="_quantized")

    # Reformat the MIDI file
    reform_mid = reformat_midi(
        mid=input_midi_path,
        verbose=verbose,
        write_to_file=False,
        override_time_info=False,
    )
    mid_tempo = get_tempo(reform_mid)

    # Convert the reformatted MIDI file to a matrix
    matrix = mid_to_matrix(reform_mid)

    # Quantize the matrix
    quantized_matrix = quantize_matrix(
        matrix=matrix, stepSize=step_size, quantizeOffsets=True, quantizeDurations=True
    )

    # Convert the quantized matrix back to a MIDI object with the specified beat resolution and velocity
    quantized_mid = matrix_to_mid(
        matrix=quantized_matrix,
        output_file=None,
        ticks_per_beat=beat_resolution,
        vel=velocity,
    )

    # Reformat the quantized MIDI object
    if mid_tempo and args.override_timing is False:
        # Set tempo to default value if tempo value was found
        final_mid = reformat_midi(
            mid=quantized_mid,
            name=output_midi_path,
            verbose=verbose,
            write_to_file=True,
            override_time_info=False,
            set_new_tempo=mid_tempo,
        )
    else:
        # Otherwise set the time
        final_mid = reformat_midi(
            mid=quantized_mid,
            name=output_midi_path,
            verbose=verbose,
            write_to_file=True,
            override_time_info=True,
            set_new_tempo=None,
        )

    return final_mid


def main():
    try:
        # Read file paths from the input data directory
        file_paths = read_files_from_directory(
            args.data_dir, extensions=args.extensions
        )

        if args.sample_size is not None and args.sample_size < len(file_paths):
            file_paths = random.sample(file_paths, args.sample_size)

        # Use Parallel and delayed to run the function in parallel
        _ = joblib.Parallel(n_jobs=args.num_cores)(
            joblib.delayed(quantize_midi_file)(file_path, args.output_dir)
            for file_path in file_paths
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
