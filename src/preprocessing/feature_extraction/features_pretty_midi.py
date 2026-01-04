"""Feature extraction utilities using the PrettyMIDI library."""

from collections import Counter

import numpy as np
import pretty_midi
from pretty_midi import (
    PrettyMIDI,
    note_number_to_hz,
    note_number_to_name,
    program_to_instrument_name,
)

from src.utils.logging.logging_manager import get_logger

logger = get_logger("features_pretty_midi")


# ++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++ INSTRUMENTATION +++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def get_midi_instruments(midi_data):
    """
    Extracts and returns a list of instrument names present in a MIDI file.

    Parameters:
    - midi_data (PrettyMIDI): PrettyMIDI object.

    Returns:
    - List[str]: A list of instrument names found in the MIDI file.

    """

    # Get a list of instruments in the MIDI file
    instruments = midi_data.instruments

    # Extract instrument names
    instrument_names = [
        program_to_instrument_name(inst.program) for inst in instruments
    ]

    instrument_names = (
        instrument_names[0] if len(instrument_names) == 1 else instrument_names
    )

    if len(instrument_names) == 0:
        logger.warning("No instrument found.")
        return 0

    return instrument_names


# ++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++ PITCH +++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def get_pitch_classes(midi_data):
    """
    Extract pitch classes from a pretty_midi PrettyMIDI object.

    Parameters:
    - midi_data: pretty_midi PrettyMIDI object

    Returns:
    - Set: Pitch classes extracted from the PrettyMIDI object.
    """
    pitch_classes = set()

    # Iterate through instruments in the PrettyMIDI object
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Calculate pitch class and add it to the set
            pitch_classes.add(note.pitch % 12)

    return list(pitch_classes)


def find_matching_scales(midi_data):
    """
    Find matching musical scales for the given pitch classes.

    Parameters:
    - midi_data: pretty_midi PrettyMIDI object

    Returns:
    - List: Matching scales for the provided pitch classes.
    """
    scale_types = {
        "ionian": [0, 2, 4, 5, 7, 9, 11],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 9, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "aeolian": [0, 2, 3, 5, 7, 8, 10],
        "locrian": [0, 2, 3, 5, 6, 8, 10],
    }

    pitch_classes = {
        0: "C",
        1: "C#",
        2: "D",
        3: "Eb",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "Ab",
        9: "A",
        10: "Bb",
        11: "B",
    }

    midi_pcs = set(get_pitch_classes(midi_data))

    matching_modes = []
    modes = scale_types.keys()
    for mode in modes:
        pc = 0
        pattern = scale_types[mode]
        while pc <= 11:
            transposed_mode = [(x + pc) % 12 for x in pattern]
            if midi_pcs.issubset(set(transposed_mode)):
                matching_modes.append(pitch_classes[pc] + " " + mode)
            pc += 1
    return matching_modes


def get_pitch_range(notes):
    """
    Calculates the pitch range of the given notes.

    Parameters:
    - notes (List[pretty_midi.Note]): A list of notes.

    Returns:
    - Tuple[float, float]: The lowest and highest frequencies in Hz and pitch values.
    """
    min_pitch = min(note.pitch for note in notes)
    max_pitch = max(note.pitch for note in notes)
    return (note_number_to_hz(min_pitch), min_pitch), (
        note_number_to_hz(max_pitch),
        max_pitch,
    )


def get_pitch_histogram(pitch_distribution):
    """
    Returns the pitch distribution as a list for histogram representation.

    Parameters:
    - pitch_distribution (dict): Dictionary containing the distribution of pitch classes.

    Returns:
    - List[int]: List containing the counts of each pitch class.
    """
    return [pitch_distribution[i] for i in range(12)]


def get_pitch_contours(notes):
    """
    Calculates the pitch contours (ascending or descending) between consecutive notes.

    Parameters:
    - notes (List[pretty_midi.Note]): A list of notes.

    Returns:
    - List[str]: A list of strings representing the contour ('ascending', 'descending', or 'same').
    """
    contours = []
    for i in range(1, len(notes)):
        if notes[i].pitch > notes[i - 1].pitch:
            contours.append("ascending")
        elif notes[i].pitch < notes[i - 1].pitch:
            contours.append("descending")
        else:
            contours.append("static")
    return contours


def get_overall_pitch_contours(directions, threshold=0.7):
    """
    Determines the overall contour of a piece based on the sequence of melodic directions.

    :param directions: A list of directions (ascending, descending, static) from the melodic_contour method
    :param threshold: A float representing the threshold to determine a predominant direction (default is 0.7)
    :return: A Tuple representing the overall contour ((1, ascending), (-1, descending), (0, mixed))
    """
    # Count the occurrences of each direction
    counts = {"ascending": 0, "descending": 0, "static": 0}
    for direction in directions:
        counts[direction] += 1

    # Calculate the proportions for each direction
    total = len(directions)

    # Handle empty directions list (avoid division by zero)
    if total == 0:
        return (0, "mixed")

    proportions = {key: value / total for key, value in counts.items()}

    # Determine the predominant direction based on the threshold
    if proportions["ascending"] >= threshold:
        return (1, "ascending")
    elif proportions["descending"] >= threshold:
        return (-1, "descending")
    else:
        return (0, "mixed")


def get_interval_analysis(midi_data, specific_sequences=None):
    """
    Analyzes the intervals between consecutive pitches in a MIDI file, including the most common intervals,
    the range of intervals, and the occurrence of specific interval sequences.

    :param midi_data: PrettyMIDI object
    :param specific_sequences: A list of specific interval sequences to search for (optional)
    :return: A dictionary containing the analysis results
    """

    # Initialize a list to store the intervals
    intervals = []

    # Iterate through the instruments and notes
    for instrument in midi_data.instruments:
        # Sort the notes by start time
        instrument.notes.sort(key=lambda note: note.start)

        # Extract the pitches
        pitches = [note.pitch for note in instrument.notes]

        # Calculate the intervals between consecutive pitches
        for i in range(1, len(pitches)):
            interval = pitches[i] - pitches[i - 1]
            intervals.append(interval)

    # Analyze the most common intervals
    most_common_intervals = Counter(intervals).most_common()

    # Determine the range of intervals (handle empty intervals)
    if intervals:
        interval_range = (min(intervals), max(intervals))
    else:
        interval_range = (None, None)

    # Search for specific interval sequences
    specific_sequence_occurrences = {}
    if specific_sequences:
        for sequence in specific_sequences:
            count = sum(
                1
                for i in range(len(intervals) - len(sequence) + 1)
                if intervals[i : i + len(sequence)] == sequence
            )
            specific_sequence_occurrences[str(sequence)] = count

    # Compile the results into a dictionary
    results = {
        "intervals": intervals,
        "most_common_intervals": most_common_intervals,
        "interval_range": interval_range,
        "specific_sequence_occurrences": specific_sequence_occurrences,
    }

    return results


def get_average_pitch(notes):
    """
    Calculates the average pitch of the given notes.

    Parameters:
    - notes (List[pretty_midi.Note]): A list of notes.

    Returns:
    - float: The average frequency in Hz.
    """
    total_pitch = sum(note_number_to_hz(note.pitch) for note in notes)
    return total_pitch / len(notes)


def get_duration_vs_pitch(notes):
    """
    Calculates the relationship between note duration and pitch.

    Parameters:
    - notes (List[pretty_midi.Note]): A list of notes.

    Returns:
    - List[Tuple[float, float]]: A list of tuples representing the duration (in seconds) and frequency (in Hz) of each note.
    """
    return [(note.end - note.start, note_number_to_hz(note.pitch)) for note in notes]


def find_overlap(midi_data):
    """
    Check if there is any overlap between notes in a pretty_midi PrettyMIDI object.

    Parameters:
    - midi_data: pretty_midi PrettyMIDI object

    Returns:
    - Boolean: True if there is overlap, False otherwise.
    """
    overlap = False
    iot = []

    # Iterate through instruments in the PrettyMIDI object
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Append the start and end times of each note to the list
            iot.append([note.start, note.end])

    # Sort the list based on the start times of the notes
    iot.sort(key=lambda x: x[0])

    # Check for overlap
    for i in range(len(iot) - 1):
        if iot[i][1] > iot[i + 1][0]:
            overlap = True
            break  # If overlap is found, no need to continue checking

    return overlap


def get_pitch_analysis(input_midi):
    """
    Extracts comprehensive pitch information from a given MIDI file, including pitch details, pitch distribution, range,
    histogram, contours, intervals, average pitch, and duration vs. pitch relationship.

    :params:
    - input_midi (str or PrettyMIDI: The path to the MIDI file or the PrettyMIDI object.

    :returns:
    - dict: A dictionary containing various pitch-related analyses, including:
        - 'pitch_information': Detailed pitch information for each note.
        - 'pitch_distribution': Distribution of pitch classes.
        - 'pitch_range': Range of pitches.
        - 'pitch_histogram': Histogram representation of pitch distribution.
        - 'pitch_contours': Contours of pitch (ascending, descending, same).
        - 'intervals': Intervals between consecutive notes.
        - 'average_pitch': Average pitch frequency.
        - 'duration_vs_pitch': Relationship between note duration and pitch.
    """
    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)

    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Initialize containers for pitch-related analyses
    notes = [note for instrument in midi_data.instruments for note in instrument.notes]
    pitch_information = []
    pitch_distribution = {i: 0 for i in range(12)}

    # Extract individual pitch information and distribution
    for note in notes:
        pitch_class = note.pitch % 12
        pitch_name = note_number_to_name(note.pitch)
        frequency = note_number_to_hz(note.pitch)
        pitch_distribution[pitch_class] += 1
        pitch_information.append(
            {
                "pitch_class": pitch_class,
                "pitch_name": pitch_name,
                "frequency": frequency,
            }
        )

    # Additional pitch-related analyses
    pitch_range = get_pitch_range(notes)
    pitch_histogram = get_pitch_histogram(pitch_distribution)
    pitch_contours = get_pitch_contours(notes)
    overall_pitch_contour = get_overall_pitch_contours(pitch_contours, threshold=0.7)
    intervals_analysis = get_interval_analysis(midi_data)
    average_pitch = get_average_pitch(notes)
    duration_vs_pitch = get_duration_vs_pitch(notes)

    return {
        "file_name": input_midi,
        "length_secs": midi_data.get_end_time(),
        "pitch_information": pitch_information,
        "pitch_distribution": pitch_distribution,
        "pitch_range": pitch_range,
        "pitch_histogram": pitch_histogram,
        "pitch_contours": pitch_contours,
        "overall_pitch_contour": overall_pitch_contour,
        "intervals_analysis": intervals_analysis,
        "average_pitch": average_pitch,
        "duration_vs_pitch": duration_vs_pitch,
    }


# ++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++ RHYTHM +++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def get_onset_times(midi_data):
    """
    Extracts the onset times of the notes from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    List[float]: List of note onset times in seconds.
    """
    onset_times = [
        note.start for instrument in midi_data.instruments for note in instrument.notes
    ]
    return sorted(onset_times)


def get_inter_onset_intervals(onset_times):
    """
    Calculates the intervals between successive note onset times.

    Parameters:
    onset_times (List[float]): List of note onset times in seconds.

    Returns:
    List[float]: List of intervals between successive note onset times.
    """
    return np.diff(onset_times).tolist()


def get_tempo_changes(midi_data):
    """
    Extracts the tempo information from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    tuple: A tuple containing arrays representing tempo changes.
    """
    # Get the tempo changes and the corresponding beat positions
    tempo_changes = midi_data.get_tempo_changes()

    # Assuming a constant tempo, take the tempo of the first event
    if tempo_changes:
        tempo = tempo_changes[1][0]

    # calculate mean tempo in case of multiple changes
    mean_tempo = 0
    if len(tempo_changes) > 2:
        mean_tempo = sum(tempo_changes) / len(tempo_changes)

    return tempo, mean_tempo, tempo_changes


def get_tempo(midi_data):
    """
    Calculate the tempo (in beats per minute) of a MIDI file based on its tick scales and resolution.

    Parameters:
    midi_data (pretty_midi.PrettyMIDI): The PrettyMIDI object representing the MIDI file.

    Returns:
    float: Tempo of the MIDI file in beats per minute.
    """
    tick_scale = midi_data._tick_scales[-1][-1]  # Extract the last tick scale
    resolution = midi_data.resolution  # Extract the resolution of the MIDI file
    beat_duration = (
        tick_scale * resolution
    )  # Calculate the duration of one beat in seconds
    mid_tempo = 60 / beat_duration  # Calculate the tempo in beats per minute
    return mid_tempo


def get_time_signature_changes(midi_data):
    """
    Extracts the time signature changes from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    List[tuple]: List of tuples representing time signature changes (numerator, denominator).
    """
    return [(ts.numerator, ts.denominator) for ts in midi_data.time_signature_changes]


def extract_energy(midi_data, fallback_tempo=120.0):
    """
    Extracts a measure of the energy from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): The PrettyMIDI object containing the MIDI data.
    fallback_tempo (float): Tempo to use if estimation fails (default: 120.0 BPM).

    Returns:
    float: A measure of the energy.
    """
    # Try to estimate tempo, with fallback strategy for sparse MIDI files
    try:
        tempo = midi_data.estimate_tempo()
    except ValueError:
        # Fallback 1: Try to get tempo from MIDI tempo changes
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes) == 2 and len(tempo_changes[1]) > 0:
            tempo = tempo_changes[1][0]  # Use first tempo change
        else:
            # Fallback 2: Use default tempo
            tempo = fallback_tempo

    dynamics = [
        np.mean([note.velocity for note in instrument.notes])
        for instrument in midi_data.instruments
    ]
    density = len(
        [note for instrument in midi_data.instruments for note in instrument.notes]
    )
    return (np.mean(dynamics) * tempo * density) / 10000.0


def extract_groove(midi_data):
    """
    Extracts a measure of groove from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): The PrettyMIDI object containing the MIDI data.

    Returns:
    float: A measure of groove, based on microtiming deviations.
    """
    onset_times = get_onset_times(midi_data)
    inter_onset_intervals = np.diff(onset_times)
    return np.std(inter_onset_intervals)


def get_rhythmic_features(input_midi):
    """
    Extracts the rhythmic characteristics from a given MIDI file.

    Parameters:
    input_midi (str orPrettyMIDI): The path to the MIDI file to analyze or the MIDI object.

    Returns:
    dict: A dictionary containing the rhythmic features, including:
          - 'onset_times': List of note onset times in seconds.
          - 'inter_onset_intervals': List of intervals between successive note onset times.
          - 'tempo_changes': List of tempo changes in the piece.
          - 'time_signature_changes': List of time signature changes in the piece.

    """

    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)

    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    onset_times = get_onset_times(midi_data)
    inter_onset_intervals = get_inter_onset_intervals(onset_times)
    mean_tempo, tempo_changes, beat_times = get_tempo_changes(midi_data)
    time_signature_changes = get_time_signature_changes(midi_data)

    return {
        "onset_times": onset_times,
        "inter_onset_intervals": inter_onset_intervals,
        "tempo_changes": tempo_changes,
        "mean_tempo": mean_tempo,
        "beat_times": beat_times,
        "time_signature_changes": time_signature_changes,
    }


def get_energy_and_groove(input_midi):
    """
    Extracts the energy and groove characteristics from a given MIDI file.

    Parameters:
    input_midi (str orPrettyMIDI): The path to the MIDI file to analyze or the MIDI object.

    Returns:
    Tuple: A tuple containing:
          - 'energy': A measure of the energy, based on dynamics, tempo, and density.
          - 'groove': A measure of groove, based on microtiming deviations.

    """

    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)

    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    energy = extract_energy(midi_data)
    groove = extract_groove(midi_data)

    return energy, groove


# ++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++ STRUCTURE +++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def extract_key_signatures(midi_data):
    """
    Extracts key signatures from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    List[dict]: List of dictionaries representing key signatures with key number and time.
    """
    key_signatures = midi_data.key_signature_changes
    return [{"key": key.key_number, "time": key.time} for key in key_signatures]


def get_number_of_bars(input_midi):
    """
    Calculate the total number of bars in a given MIDI file.

    Parameters:
    - midi_file_path (str): Path to the input MIDI file.

    Returns:
    - int: Total number of bars in the MIDI file.
    """

    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Get time signature changes
    time_signatures = midi_data.time_signature_changes
    if not time_signatures:
        # Default to 4/4 if no time signature is specified
        time_signatures = [pretty_midi.TimeSignature(4, 4, 0)]

    total_bars = 0
    for i, ts in enumerate(time_signatures):
        # If this isn't the last time signature change, then we compute the
        # number of bars between this change and the next one. Otherwise,
        # compute the number of bars till the end of the track.
        start_time = ts.time
        end_time = (
            time_signatures[i + 1].time
            if i + 1 < len(time_signatures)
            else midi_data.get_end_time()
        )

        # Get all beats and filter them between start_time and end_time
        all_beats = midi_data.get_beats()
        beats = [beat for beat in all_beats if start_time <= beat < end_time]

        # Calculate number of bars between start_time and end_time using the current time signature
        bars = len(beats) / ts.numerator
        total_bars += bars

    return np.ceil(total_bars)


def extract_time_signatures(midi_data):
    """
    Extracts time signatures from the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    List[dict]: List of dictionaries representing time signatures with numerator, denominator, and time.
    """
    time_signatures = midi_data.time_signature_changes
    return [
        {"numerator": ts.numerator, "denominator": ts.denominator, "time": ts.time}
        for ts in time_signatures
    ]


def extract_instrumentation(midi_data):
    """
    Extracts the names of instruments used in the given MIDI data.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.

    Returns:
    List[str]: List of instrument names used in the MIDI, excluding drums.
    """
    instrumentation = []
    for instr in midi_data.instruments:
        if not instr.is_drum:
            program_number = int(instr.program)
            instrument_name = program_to_instrument_name(program_number)
            instrumentation.append(instrument_name)
    return instrumentation


def determine_orchestration(midi_data, sampling_time):
    """
    Determines the orchestration by sampling the MIDI data at regular intervals to find active instruments.

    Parameters:
    midi_data (PrettyMIDI): MIDI data to analyze.
    sampling_time (float): Time interval in seconds at which the MIDI is sampled.

    Returns:
    List[List[str]]: List of active instruments at each sampled time point, excluding drums.
    """
    time_points = [
        i * sampling_time for i in range(int(midi_data.get_end_time() / sampling_time))
    ]
    orchestration = []

    for time in time_points:
        active_instruments = []
        for instr in midi_data.instruments:
            if not instr.is_drum:
                for note in instr.notes:
                    if note.start <= time <= note.end:
                        active_instruments.append(
                            program_to_instrument_name(instr.program)
                        )
                        break
        orchestration.append(active_instruments)

    return orchestration


def get_structure_information(input_midi, sampling_time=0.1):
    """
    Extracts the structural information from a given MIDI file or a PrettyMIDI object.
    The extracted information includes key signatures, time signatures, tempo changes,
    instrumentation, and orchestration.

    Parameters:
        input_midi (str or PrettyMIDI): The input MIDI data, either as a file path (str)
            or a PrettyMIDI object.
        sampling_time (float, optional): The time interval in seconds at which the MIDI is
            sampled to determine orchestration. Default is 0.1 seconds.

    Returns:
        dict: A dictionary containing the extracted structural information, including:
            - 'key_signatures': List of key signatures with key number and time.
            - 'time_signatures': List of time signatures with numerator, denominator, and time.
            - 'tempo_changes': A tuple containing an array of tempo changes and corresponding times.
            - 'instrumentation': List of instrument names used in the MIDI.
            - 'orchestration': List of active instruments at each sampled time point.

    Raises:
        TypeError: If the input is neither a path to a MIDI file (str) nor a PrettyMIDI object.
    """

    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    keys = extract_key_signatures(midi_data)
    times = extract_time_signatures(midi_data)
    tempos = get_tempo_changes(midi_data)
    instrumentation = extract_instrumentation(midi_data)
    orchestration = determine_orchestration(midi_data, sampling_time)

    return {
        "key_signatures": keys,
        "time_signatures": times,
        "tempo_changes": tempos,
        "instrumentation": instrumentation,
        "orchestration": orchestration,
    }


# ++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++ HARMONY ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def extract_harmonic_content(input_midi):
    """
    Extracts the harmonic content information from a given MIDI file.

    The method calculates the pitch class histogram, which represents the
    distribution of pitch classes (notes) within the piece.

    Parameters:
        input_midi (PrettyMIDI): PrettyMIDI object.

    Returns:
        pitch_class_histogram (list): A list containing 12 elements,
            representing the distribution of the 12 pitch classes (C, C#, D, ..., B)
            in the MIDI file.

    Example:
        pitch_class_histogram = extract_harmonic_content('path/to/midi/file.mid')
        print(pitch_class_histogram)  # [0.1, 0.05, 0.15, ..., 0.05]

    Note:
        The values in the pitch_class_histogram are normalized, so they sum to 1.
        Each value represents the proportion of that particular pitch class in the piece.
    """
    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Initialize the pitch class histogram with zeros
    pitch_class_histogram = [0] * 12

    # Iterate through all the instruments and notes in the MIDI file
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Increment the corresponding pitch class by the note's duration
            pitch_class_histogram[note.pitch % 12] += note.end - note.start

    # Normalize the histogram so that the sum is 1
    total_duration = sum(pitch_class_histogram)
    if total_duration > 0:
        pitch_class_histogram = [
            count / total_duration for count in pitch_class_histogram
        ]

    return pitch_class_histogram


def extract_melody_harmony_relationship(input_midi):
    """
    Extracts the melody and harmony relationship from a given MIDI file.

    Parameters:
        midi_file_path (str): The path to the MIDI file.

    Returns:
        relationship_info (dict): A dictionary containing:
            - 'melodic_line': A list representing the main melodic line (pitch sequence).
            - 'harmonic_intervals': A list of harmonic intervals accompanying the melody.
            - 'chord_progression': A list of chords detected in the harmony.

    Example:
        relationship_info = extract_melody_harmony_relationship('path/to/midi/file.mid')
        print(relationship_info['melodic_line'])  # [60, 62, 64, ...]
    """
    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Identify the main melodic line (e.g., the instrument with the most notes)
    melodic_instrument = max(midi_data.instruments, key=lambda x: len(x.notes))
    melodic_line = [note.pitch for note in melodic_instrument.notes]

    # Analyze the harmonic content by looking at the other instruments
    harmonic_intervals = []
    chord_progression = []

    # Time step for analyzing harmony (can be adjusted as needed)
    time_step = 0.5
    melodic_index = 0

    # Iterate through the time, analyzing the harmony
    for time in range(int(midi_data.get_end_time() / time_step)):
        notes_at_time = []
        for instrument in midi_data.instruments:
            if instrument != melodic_instrument:
                for note in instrument.notes:
                    if note.start <= time * time_step < note.end:
                        notes_at_time.append(note.pitch)

        # Check if there's a melodic note at this time step
        if (
            melodic_index < len(melodic_instrument.notes)
            and melodic_instrument.notes[melodic_index].start
            <= time * time_step
            < melodic_instrument.notes[melodic_index].end
        ):
            melodic_note = melodic_line[melodic_index]
            intervals = [abs(n - melodic_note) % 12 for n in notes_at_time]
            harmonic_intervals.append(intervals)
        else:
            harmonic_intervals.append([])

        # Move to the next melodic note if the current one has ended
        if (
            melodic_index < len(melodic_instrument.notes)
            and time * time_step >= melodic_instrument.notes[melodic_index].end
        ):
            melodic_index += 1

        # Detect chords (assuming a simple triad structure)
        if len(notes_at_time) >= 3:
            chord_notes = sorted(notes_at_time)[:3]
            chord = (
                pretty_midi.note_number_to_name(chord_notes[0])
                + "-"
                + pretty_midi.note_number_to_name(chord_notes[1])
                + "-"
                + pretty_midi.note_number_to_name(chord_notes[2])
            )
            chord_progression.append(chord)

    # Compile relationship information
    relationship_info = {
        "melodic_line": melodic_line,
        "harmonic_intervals": harmonic_intervals,
        "chord_progression": chord_progression,
    }

    return relationship_info


# ++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++ EXPRESSIVENESS ++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def extract_average_velocity(instrument):
    """
    Extracts the average velocity for a given instrument.

    Parameters:
    instrument (PrettyMIDI Instrument): Instrument to analyze.

    Returns:
    float: The average velocity for the given instrument.
    """
    velocities = [note.velocity for note in instrument.notes]
    return sum(velocities) / len(velocities) if velocities else 0


def extract_timing_deviations(instrument):
    """
    Extracts the timing deviations for a given instrument.

    Parameters:
    instrument (PrettyMIDI Instrument): Instrument to analyze.

    Returns:
    List[float]: The timing deviations for the given instrument.
    """
    return [
        note.start - instrument.notes[i - 1].start
        for i, note in enumerate(instrument.notes)
        if i > 0
    ]


def extract_articulations(instrument):
    """
    Extracts the articulations for a given instrument.

    Parameters:
    instrument (PrettyMIDI Instrument): Instrument to analyze.

    Returns:
    List[float]: The articulations for the given instrument.
    """
    return [
        instrument.notes[i + 1].start - note.end
        for i, note in enumerate(instrument.notes)
        if i < len(instrument.notes) - 1
    ]


def get_performance_expressiveness(input_midi):
    """
    Extracts the performance expressiveness information from a given MIDI file.

    Performance expressiveness is characterized by dynamics (velocity),
    timing deviations, and articulations.

    Parameters:
        input_midi (str or PrettyMIDI): Midi file path or PrettyMIDI object.

    Returns:
        expressiveness_info (dict): A dictionary containing the following keys:
            - 'average_velocity': A list of average velocities for each instrument.
            - 'timing_deviations': A list of timing deviations for each instrument.
            - 'articulations': A list of articulations for each instrument.

    Example:
        expressiveness_info = extract_performance_expressiveness('path/to/midi/file.mid')
        print(expressiveness_info['average_velocity'])  # [64, 72, ...]
    """
    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Initialize the expressiveness information
    expressiveness_info = {
        "average_velocity": [],
        "timing_deviations": [],
        "articulations": [],
    }

    for instrument in midi_data.instruments:
        average_velocity = extract_average_velocity(instrument)
        timing_deviations = extract_timing_deviations(instrument)
        articulations = extract_articulations(instrument)

        expressiveness_info["average_velocity"].append(average_velocity)
        expressiveness_info["timing_deviations"].append(timing_deviations)
        expressiveness_info["articulations"].append(articulations)

    return expressiveness_info


# ++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++ TEXTURE ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++


def extract_texture_and_polyphony(input_midi, time_step=0.1):
    """
    Extracts the texture and polyphony information from a given MIDI file.

    Texture is characterized by the interaction of multiple voices or instruments.
    Polyphony involves the simultaneous sounding of independent melodic lines.

    Parameters:
        input_midi (str or PrettyMIDI): Midi file path or PrettyMIDI object.
        time_step (float): ime step for counting simultaneous notes.

    Returns:
        texture_info (dict): A dictionary containing the following keys:
            - 'max_polyphony': The maximum number of simultaneous notes.
            - 'average_polyphony': The average number of simultaneous notes.
            - 'instrument_count': The number of instruments.
            - 'note_density': The total number of notes divided by the total duration.

    Example:
        texture_info = extract_texture_and_polyphony('path/to/midi/file.mid')
        print(texture_info['max_polyphony'])  # 5
    """
    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        midi_data = PrettyMIDI(input_midi)
    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        midi_data = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Initialize variables for polyphony counts and total duration
    polyphony_counts = []
    total_duration = midi_data.get_end_time()

    # Iterate through the time, counting simultaneous notes
    for time in range(int(total_duration / time_step)):
        count = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start <= time * time_step < note.end:
                    count += 1
        polyphony_counts.append(count)

    # Calculate texture information
    texture_info = {
        "max_polyphony": max(polyphony_counts),
        "average_polyphony": sum(polyphony_counts) / len(polyphony_counts),
        "instrument_count": len(midi_data.instruments),
        "note_density": sum(
            len(instrument.notes) for instrument in midi_data.instruments
        )
        / total_duration,
    }

    return texture_info
