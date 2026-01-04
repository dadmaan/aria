"""MIDI quantization and matrix conversion utilities."""

import math
import os

import pandas as pd
from mido import Message, MetaMessage, MidiFile, MidiTrack

from src.utils.logging.logging_manager import get_logger

logger = get_logger("quantization")


def parse_mid(mid):
    """
    This function allows to pass midi files or streams
    without redundant reloading files from disk.

    """
    if type(mid) is str:
        return MidiFile(mid)

    elif type(mid) is MidiFile:
        return mid

    else:
        raise (IOError("Not valid midi file or path."))


def mid_to_matrix(mid, output="nested_list"):
    """
    Takes a MIDI file or stream and returns a matrix with rows representing MIDI events
    and columns representing MIDI note, offset position, event duration, velocity, and channel:

    |      | pitch | offset | duration | velocity | channel |
    | ---- | ----- | ------ | -------- | -------- | ------- |
    | evt1 |       |        |          |          |         |
    | evt2 |       |        |          |          |         |
    | evt3 |       |        |          |          |         |
    | ...  |       |        |          |          |         |

    Parameters:
    - mid: MIDI file or stream to be converted to a matrix.
    - output (str, optional): Output format of the matrix. Can be 'nested_list' or 'pandas'.
                             Defaults to 'nested_list'.

    Returns:
    - list or pandas.DataFrame: The matrix representing MIDI events. If 'nested_list' is chosen,
                                a list of lists is returned. If 'pandas' is chosen, a
                                pandas DataFrame is returned with columns: ['pitch', 'offset', 'duration'].

    Notes:
    - This function converts a MIDI file or stream into a matrix format.
    - The matrix has rows representing MIDI events and columns representing MIDI note,
      offset position, event duration, velocity, and channel.
    - The `output` parameter determines the format of the returned matrix ('nested_list' or 'pandas').
    - If the MIDI file has a type other than 0, a message is printed, and the function returns None.
    - The function extracts note-on and note-off events, calculates their durations, and constructs
      the matrix with MIDI note, offset, and duration information.
    - If the lengths of note-on and note-off events do not match, a message is printed, and the function returns None.

    Example:
    >>> midi_file = "example.mid"
    >>> matrix = mid_to_matrix(midi_file, output='nested_list')
    >>> print("MIDI file converted to matrix successfully.")
    """
    # Parse the MIDI file or stream
    mid = parse_mid(mid)

    # Check if the MIDI file type is not 0
    if mid.type != 0:
        logger.warning(
            "MIDI file type {}. Reformat to type 0 before quantizing.".format(mid.type)
        )
        return None

    # Get the resolution of the MIDI file
    resolution = mid.ticks_per_beat

    # Initialize variables to store MIDI events
    elapsed = 0
    noteons = []
    offsets = []
    noteoffs = []
    durations = []

    # Extract note-on and note-off events from the MIDI file
    for msg in mid.tracks[0]:
        elapsed += msg.time
        offset = elapsed / resolution
        if msg.type == "note_on":
            noteons.append(msg.note)
            offsets.append(offset)
        if msg.type == "note_off":
            noteoffs.append(msg.note)
            durations.append(offset)

    # Check if the lengths of note-on and note-off events match
    if not len(noteons) == len(noteoffs):
        logger.error("Mismatched size. Reformat file first.")
        return None
    else:
        # Construct the matrix with MIDI note, offset, and duration information
        mnotes = []
        for i in range(len(noteons)):
            mnotes.append(
                [
                    noteons[i],
                    offsets[i],
                    durations[noteoffs.index(noteons[i])] - offsets[i],
                ]
            )
            durations.pop(noteoffs.index(noteons[i]))
            noteoffs.remove(noteons[i])

        # Return the matrix in the specified format
        if output == "nested_list":
            return mnotes
        elif output == "pandas":
            return pd.DataFrame(mnotes, columns=["pitch", "offset", "duration"])


def quantize_matrix(
    matrix, stepSize=0.25, quantizeOffsets=True, quantizeDurations=True
):
    """
    Quantize a note matrix to fit the desired grid.

    Parameters:
    - matrix (list): A matrix containing MIDI events. Each row of the matrix
                    may have three elements: [pitch, start_time, duration].
    - stepSize (float, optional): The quantization factor in multiples or fractions
                                 of quarter notes. Defaults to 0.25.
    - quantizeOffsets (bool, optional): If True, adjust note start times to the grid.
                                       Defaults to True.
    - quantizeDurations (bool, optional): If True, adjust note durations to the grid.
                                         Defaults to True.

    Returns:
    list: The quantized note matrix.

    Notes:
    - This function quantizes a note matrix based on the specified parameters.
    - The `stepSize` parameter determines the granularity of the quantization grid.
    - If `quantizeOffsets` is True, note start times are adjusted to the nearest grid point.
    - If `quantizeDurations` is True, note durations are adjusted to the nearest grid point.
    - The quantized note matrix is returned.

    Example:
    >>> input_matrix = [[60, 1.1, 0.3], [62, 2.3, 0.4], [64, 3.8, 0.2]]
    >>> quantized_matrix = quantize_matrix(input_matrix, stepSize=0.5, quantizeOffsets=True, quantizeDurations=True)
    >>> print("Note matrix quantized successfully.")
    """

    # Calculate the number of grid points per quarter note
    beat_grid = 2 * (1.0 / stepSize)

    # Iterate through each event in the matrix
    for e in matrix:
        # Quantize note start times if enabled
        if quantizeOffsets:
            starts = (e[1] * beat_grid) % 2
            if starts < 1.0:
                e[1] = math.floor(e[1] * beat_grid) / beat_grid  # Round down
            elif starts == 1.0:
                e[1] = (
                    (e[1] - (stepSize * 0.5)) * beat_grid
                ) / beat_grid  # Adjust for midpoint
            else:
                e[1] = math.ceil(e[1] * beat_grid) / beat_grid  # Round up

        # Quantize note durations if enabled
        if quantizeDurations:
            if e[2] < (stepSize * 0.5):
                e[2] = stepSize  # Minimum duration
            else:
                durs = (e[2] * beat_grid) % 2
                if durs < 1.0:
                    e[2] = math.floor(e[2] * beat_grid) / beat_grid  # Round down
                elif durs == 1.0:
                    e[2] = (
                        (e[2] + (stepSize * 0.5)) * beat_grid
                    ) / beat_grid  # Adjust for midpoint
                else:
                    e[2] = math.ceil(e[2] * beat_grid) / beat_grid  # Round up

    return matrix


def matrix_to_mid(matrix, output_file=None, ticks_per_beat=480, vel=100):
    """
    Convert a matrix representation of musical events to a MIDI file.

    Parameters:
    - matrix (list): A 2D matrix representing musical events. Each row of the matrix
                    may have three elements: [pitch, start_time, duration].
    - output_file (str, optional): The path to save the generated MIDI file. If not
                                   provided, the MIDI file will not be saved.
    - ticks_per_beat (int, optional): The number of ticks per beat in the generated MIDI file.
                                      Defaults to 480, a common value in many MIDI files.
    - vel (int, optional): The velocity (volume) of the notes. Defaults to 100.

    Returns:
    pretty_midi.PrettyMIDI: The generated PrettyMIDI object representing the MIDI file.

    Notes:
    - This function takes a matrix of musical events and converts it into a MIDI file.
    - Each row in the matrix should represent a musical event with [pitch, start_time, duration].
    - The generated MIDI file is saved to the specified output file if `output_file` is provided.
    - The `ticks_per_beat` parameter determines the timing resolution of the MIDI file.
    - The `vel` parameter sets the default velocity for note events.
    - The MIDI file is structured with a single track containing note-on and note-off events.
    - Note events are sorted by their start times before being added to the MIDI track.

    Example:
    >>> matrix = [[60, 0, 1], [62, 1, 0.5], [64, 1.5, 0.5]]
    >>> midi_object = matrix_to_mid(matrix, output_file="output.mid", ticks_per_beat=480, vel=100)
    >>> print("MIDI file generated successfully.")
    """
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat
    mid.type = 0
    track = MidiTrack()
    mid.tracks.append(track)

    if output_file is not None:
        track.append(
            MetaMessage("track_name", name=os.path.split(output_file)[1], time=int(0))
        )
        track.append(MetaMessage("set_tempo", tempo=480000, time=int(0)))
        track.append(
            MetaMessage("time_signature", numerator=4, denominator=4, time=int(0))
        )

    # Combine note-on and note-off events, then sort by their time
    sort_events = []
    for row in matrix:
        sort_events.append([row[0], 1, row[1]])
        sort_events.append([row[0], 0, (row[1] + row[2])])

    sort_events.sort(key=lambda tup: tup[2])

    # Convert sorted events to MIDI events
    lapso = 0
    for evt in sort_events:
        if evt[1] == 1:
            track.append(
                Message(
                    "note_on",
                    note=evt[0],
                    velocity=vel,
                    time=int((evt[2] - lapso) * ticks_per_beat),
                )
            )
            lapso = evt[2]
        elif evt[1] == 0:
            track.append(
                Message(
                    "note_off",
                    note=evt[0],
                    velocity=0,
                    time=int((evt[2] - lapso) * ticks_per_beat),
                )
            )
            lapso = evt[2]

    if output_file is not None:
        track.append(MetaMessage("end_of_track", time=(int(0))))
        mid.save(output_file)

    return mid


def reformat_midi(
    mid,
    name=None,
    verbose=True,
    write_to_file=False,
    override_time_info=False,
    set_new_tempo=None,
):
    """
    Performs sanity check and reformats a midi file based on the following criteria:

    - Flattens all messages onto a single track, making it of midi file type 0.
    - Converts 'note_on' messages with velocity=0 to 'note_off' messages.
    - Checks if the last 'note_on' has a corresponding 'note_off' message, adding one if needed.
    - Adds an 'end_of_track' metamessage that is a multiple of the time_signature.

    Reformatting will make the file load better (i.e. nicer looking) in music21 and other musicxml programs.

    Parameters
    ----------
    mid: str or mido.MidiFile:
        Valid path to a midi file or midi stream.
    name: str
        different name...
    verbose: bool
        Print messages to the console while formatting
    write_to_file: bool
        Overwrite the original midi file with the newly formatted data.
    override_time_info: bool
        Override original tempo to 120 and time signature to 4/4.
    set_new_tempo: int
        Override original tempo with new tempo.

    Return
    ------
    mid: mido.MidiFile
        A pythonised midi file for further manipulation.

    Notes
    -----
    override_time_info ignores the original tempo and time signature,
    forcing them to 'set_tempo' = 125 bmp's and 'time_signature' = 4/4.
    This is useful for most cases of analysis of EDM content.

    """

    mid = parse_mid(mid)

    if not mid.filename:
        mid.filename = "midi_track"

    if not name:
        name = os.path.join(os.getcwd(), mid.filename)

    logger.info("file name: {}".format(mid.filename))

    if verbose:
        logger.info("file type: {}".format(mid.type))
        logger.info("ticks per quarter note: {}".format(mid.ticks_per_beat))
        logger.info("number of tracks {}".format(len(mid.tracks)))
        logger.info(str(mid.tracks))

    EXCLUDED_MSG_TYPES = {
        "sequence_number",
        "text",
        "copyright",
        "track_name",
        "instrument_name",
        "lyrics",
        "marker",
        "cue_marker",
        "device_name",
        "channel_prefix",
        "midi_port",
        "sequencer_specific",
        "end_of_track",
        "smpte_offset",
    }

    if override_time_info:
        EXCLUDED_MSG_TYPES.add("time_signature")
        EXCLUDED_MSG_TYPES.add("set_tempo")

    if set_new_tempo:
        EXCLUDED_MSG_TYPES.add("time_signature")
        EXCLUDED_MSG_TYPES.add("set_tempo")

    # if type 2, do nothing!
    if mid.type == 2:
        logger.warning(
            "Midi file type {}. I did not dare to change anything.".format(mid.type)
        )
        return None

    else:
        if verbose and mid.type == 1:
            # if type 1, convert to type 0
            logger.info("Converting file type 1 to file type 0 (single track).")

        flat_track = MidiTrack()
        flat_track.append(
            MetaMessage("track_name", name=os.path.split(name)[1], time=0)
        )
        logger.debug("NAME {}".format(os.path.split(name)[1]))
        flat_track.append(MetaMessage("track_name", name="unnamed", time=0))
        flat_track.append(MetaMessage("instrument_name", name="Bass", time=0))

        if override_time_info:
            if verbose:
                logger.warning("Ignoring Tempo and Time Signature Information.")
            flat_track.append(MetaMessage("set_tempo", tempo=500_000, time=0))
            flat_track.append(
                MetaMessage("time_signature", numerator=4, denominator=4, time=0)
            )

        # Set the new tempo in the tempo track
        if set_new_tempo:
            if verbose:
                logger.warning("Setting New Tempo Information.")
            flat_track.append(MetaMessage("set_tempo", tempo=set_new_tempo, time=0))
            flat_track.append(
                MetaMessage("time_signature", numerator=4, denominator=4, time=0)
            )

        for track in mid.tracks:
            for msg in track:
                if any(msg.type == msg_type for msg_type in EXCLUDED_MSG_TYPES):
                    if verbose:
                        logger.debug("IGNORING {}".format(msg))
                else:
                    flat_track.append(msg)

        # replace the 'tracks' field with a single track containing all the messages.
        # later on we can check for duplicates in certain fields (tempo, timesignature, key)
        mid.tracks.clear()
        mid.type = 0
        mid.tracks.append(flat_track)

    # Convert 'note_on' messages with velocity 0 to note_off messages:
    for msg in mid.tracks[0]:
        if msg.type == "note_on" and msg.velocity == 0:
            if verbose:
                logger.info(
                    "Replacing 'note_on' with velocity=0 with a 'note_off' message (track[{}])".format(
                        mid.tracks[0].index(msg)
                    )
                )
            mid.tracks[0].insert(
                mid.tracks[0].index(msg),
                Message(
                    "note_off", note=msg.note, velocity=msg.velocity, time=msg.time
                ),
            )
            mid.tracks[0].remove(msg)

    # Add a 'note_off' event at the end of track if it were missing:
    events = []
    for msg in mid.tracks[0]:
        if msg.type == "note_on" or msg.type == "note_off":
            events.append(msg)
    if len(events) > 0:
        if events[-1].type == "note_on":
            mid.tracks[0].append(
                Message("note_off", note=events[-1].note, velocity=0, time=0)
            )
            if verbose:
                logger.warning(
                    "'note_off' missing at the end of file. Adding 'note_off' message."
                )

    # Set the duration of the file to a multiple of the Time Signature:
    ticks_per_beat = mid.ticks_per_beat
    beats_per_bar = 4
    dur_in_ticks = 0
    for msg in mid.tracks[0]:
        dur_in_ticks += msg.time
        if msg.type == "set_tempo":
            if verbose:
                logger.info("Tempo: {} BPM".format(60_000_000 / msg.tempo))
        if msg.type == "time_signature":
            beats_per_bar = msg.numerator
            ticks_per_beat = (4 / msg.denominator) * mid.ticks_per_beat
            if verbose:
                logger.info(
                    "Time Signature: {}/{}".format(msg.numerator, msg.denominator)
                )

    ticks_per_bar = beats_per_bar * ticks_per_beat
    dur_in_measures = dur_in_ticks / ticks_per_bar
    expected_dur_in_ticks = int(math.ceil(dur_in_measures) * ticks_per_bar)
    ticks_to_end_of_bar = expected_dur_in_ticks - dur_in_ticks
    logger.debug("ticks_to_end_of_bar: {}".format(ticks_to_end_of_bar))

    if mid.tracks[0][-1].type == "end_of_track":
        ticks_to_end_of_bar += mid.tracks[0][-1].time
        mid.tracks[0].pop(-1)

    mid.tracks[0].append(MetaMessage("end_of_track", time=ticks_to_end_of_bar))

    if verbose:
        if dur_in_ticks == expected_dur_in_ticks:
            logger.info("Original duration already a multiple of Time Signature.")
            logger.info("{} ticks, {} bars.".format(dur_in_ticks, dur_in_measures))
        else:
            logger.info(
                "Original duration: {} ticks, {} bars.".format(
                    dur_in_ticks, dur_in_measures
                )
            )
            new_dur_in_ticks = 0
            for msg in mid.tracks[0]:
                new_dur_in_ticks += msg.time
            logger.info(
                "Final duration: {} ticks, {} bars.".format(
                    new_dur_in_ticks, new_dur_in_ticks / ticks_per_bar
                )
            )

    if write_to_file:
        mid.save(name)
        if verbose:
            logger.info("(Over)writting mid file with changes.\n")

    return mid
