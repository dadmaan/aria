"""MIDI utilities for instrument handling and audio processing."""

import os
import random
from copy import deepcopy

import numpy as np
from pretty_midi import Instrument, Note, PrettyMIDI

from src.utils.logging.logging_manager import get_logger

logger = get_logger("midi_utils")


def _get_audio_backend():
    """Return the optional IPython Audio callable or raise a helpful error."""

    if Audio is None:
        raise RuntimeError(
            "IPython.display.Audio is unavailable. Install 'ipython' or "
            "provide a custom audio playback handler before calling this function."
        )

    return Audio


def get_midi_class_instruments(instrument_class):
    """
    Given the name of a MIDI instrument, return the corresponding MIDI program change number.

    :parame: instrument_name (str): The name of the MIDI instrument.

    :returns: int: The MIDI program change number for the given instrument,
                    or None if the instrument is not found.
    """
    midi_instruments = {
        "piano": [
            (0, "Acoustic Grand Piano"),
            (1, "Bright Acoustic Piano"),
            (2, "Electric Grand Piano"),
            (3, "Honky-tonk Piano"),
            (4, "Rhodes Piano"),
            (5, "Chorused Piano"),
            (6, "Harpsichord"),
            (7, "Clavinet"),
        ],
        "chromatic percussion": [
            (8, "Celesta"),
            (9, "Glockenspiel"),
            (10, "Music box"),
            (11, "Vibraphone"),
            (12, "Marimba"),
            (13, "Xylophone"),
            (14, "Tubular Bells"),
            (15, "Dulcimer"),
        ],
        "organ": [
            (16, "Hammond Organ"),
            (17, "Percussive Organ"),
            (18, "Rock Organ"),
            (19, "Church Organ"),
            (20, "Reed Organ"),
            (21, "Accordion"),
            (22, "Harmonica"),
            (23, "Tango Accordion"),
        ],
        "guitar": [
            (24, "Acoustic Guitar (nylon)"),
            (25, "Acoustic Guitar (steel)"),
            (26, "Electric Guitar (jazz)"),
            (27, "Electric Guitar (clean)"),
            (28, "Electric Guitar (muted)"),
            (29, "Overdriven Guitar"),
            (30, "Distortion Guitar"),
            (31, "Guitar Harmonics"),
        ],
        "bass": [
            (32, "Acoustic Bass"),
            (33, "Electric Bass (finger)"),
            (34, "Electric Bass (pick)"),
            (35, "Fretless Bass"),
            (36, "Slap Bass 1"),
            (37, "Slap Bass 2"),
            (38, "Synth Bass 1"),
            (39, "Synth Bass 2"),
        ],
        "strings": [
            (40, "Violin"),
            (41, "Viola"),
            (42, "Cello"),
            (43, "Contrabass"),
            (44, "Tremolo Strings"),
            (45, "Pizzicato Strings"),
            (46, "Orchestral Harp"),
            (47, "Timpani"),
        ],
        "ensemble": [
            (48, "String Ensemble 1"),
            (49, "String Ensemble 2"),
            (50, "Synth Strings 1"),
            (51, "Synth Strings 2"),
            (52, "Choir Aahs"),
            (53, "Voice Oohs"),
            (54, "Synth Voice"),
            (55, "Orchestra Hit"),
        ],
        "brass": [
            (56, "Trumpet"),
            (57, "Trombone"),
            (58, "Tuba"),
            (59, "Muted Trumpet"),
            (60, "French Horn"),
            (61, "Brass Section"),
            (62, "Synth Brass 1"),
            (63, "Synth Brass 2"),
        ],
        "reed": [
            (64, "Soprano Sax"),
            (65, "Alto Sax"),
            (66, "Tenor Sax"),
            (67, "Baritone Sax"),
            (68, "Oboe"),
            (69, "English Horn"),
            (70, "Bassoon"),
            (71, "Clarinet"),
        ],
        "pipe": [
            (72, "Piccolo"),
            (73, "Flute"),
            (74, "Recorder"),
            (75, "Pan Flute"),
            (76, "Bottle Blow"),
            (77, "Shakuhachi"),
            (78, "Whistle"),
            (79, "Ocarina"),
        ],
        "synth lead": [
            (80, "Lead 1 (square)"),
            (81, "Lead 2 (sawtooth)"),
            (82, "Lead 3 (calliope lead)"),
            (83, "Lead 4 (chiffer lead)"),
            (84, "Lead 5 (charang)"),
            (85, "Lead 6 (voice)"),
            (86, "Lead 7 (fifths)"),
            (87, "Lead 8 (brass + lead)"),
        ],
        "synth pad": [
            (88, "Pad 1 (new age)"),
            (89, "Pad 2 (warm)"),
            (90, "Pad 3 (polysynth)"),
            (91, "Pad 4 (choir)"),
            (92, "Pad 5 (bowed)"),
            (93, "Pad 6 (metallic)"),
            (94, "Pad 7 (halo)"),
            (95, "Pad 8 (sweep)"),
        ],
        "synth effects": [
            (96, "FX 1 (rain)"),
            (97, "FX 2 (soundtrack)"),
            (98, "FX 3 (crystal)"),
            (99, "FX 4 (atmosphere)"),
            (100, "FX 5 (brightness)"),
            (101, "FX 6 (goblins)"),
            (102, "FX 7 (echoes)"),
            (103, "FX 8 (sci-fi)"),
        ],
        "ethnic": [
            (104, "Sitar"),
            (105, "Banjo"),
            (106, "Shamisen"),
            (107, "Koto"),
            (108, "Kalimba"),
            (109, "Bagpipe"),
            (110, "Fiddle"),
            (111, "Shana"),
        ],
        "percussive": [
            (112, "Tinkle Bell"),
            (113, "Agogo"),
            (114, "Steel Drums"),
            (115, "Woodblock"),
            (116, "Taiko Drum"),
            (117, "Melodic Tom"),
            (118, "Synth Drum"),
            (119, "Reverse Cymbal"),
        ],
        "sound effects": [
            (120, "Guitar Fret Noise"),
            (121, "Breath Noise"),
            (122, "Seashore"),
            (123, "Bird Tweet"),
            (124, "Telephone Ring"),
            (125, "Helicopter"),
            (126, "Applause"),
            (127, "Gunshot"),
        ],
    }

    return midi_instruments.get(instrument_class, "Instrument class not found")


def get_midi_paths(directory):
    """
    Processes the given directory and returns a list of MIDI files.

    :param midi_path: the path to the MIDI file to process
    :return: the list containing the path to the MIDI files
    """
    midi_files = []

    # Walk through all files in the directory that includes the sub-directories as well
    for dirpath, _dirnames, filenames in os.walk(directory):
        for filename in [
            f for f in filenames if f.endswith(".midi") or f.endswith(".mid")
        ]:
            # Add the file path in the list
            midi_files.append(os.path.join(dirpath, filename))

    return midi_files


def play_midi(midi_path: str, fs=22050):
    """
    Play a PrettyMIDI object.

    :param pretty_midi.PrettyMIDI pm: The PrettyMIDI object.
    :param int fs: The sampling rate for the audio waveform.

    :return: An IPython.display.Audio object that can play the audio.
    """

    pm = PrettyMIDI(midi_path)

    # Synthesize the PrettyMIDI object to an audio waveform
    audio_data = pm.synthesize(fs=fs)

    # Normalize the audio waveform to lie between -1 and 1
    audio_data = audio_data / np.abs(audio_data).max()

    audio_backend = _get_audio_backend()

    # Return an Audio object that can play the audio
    return audio_backend(audio_data, rate=fs)


def play_pretty_midi(pm: PrettyMIDI, fs=22050):
    """
    Play a PrettyMIDI object.

    :param pretty_midi.PrettyMIDI pm: The PrettyMIDI object.
    :param int fs: The sampling rate for the audio waveform.

    :return: An IPython.display.Audio object that can play the audio.
    """

    # Synthesize the PrettyMIDI object to an audio waveform
    audio_data = pm.synthesize(fs=fs)

    # Normalize the audio waveform to lie between -1 and 1
    audio_data = audio_data / np.abs(audio_data).max()

    audio_backend = _get_audio_backend()

    # Return an Audio object that can play the audio
    return audio_backend(audio_data, rate=fs)


def midi_to_piano_roll(midi_path):
    """
    Converts a MIDI file to a binary piano roll matrix.

    This function reads a MIDI file and returns a binary piano roll matrix.
    The rows of the matrix represent different pitches (from 0 to 127, which is
    the range of pitches that a MIDI file can represent), and the columns represent
    different time steps. Each time step is 1/100th of a second, which is the default
    resolution for the `get_piano_roll` method.

    If the MIDI file contains multiple instruments, this function will merge their piano
    rolls together.

    :params: midi_path (str): Path to the MIDI file.

    :returns: numpy.ndarray: Binary piano roll matrix representation.
    """
    # Load MIDI file into PrettyMIDI object
    pm = PrettyMIDI(midi_path)

    # Get piano roll of the MIDI file, this method returns a binary matrix,
    # where each row corresponds to a different pitch and each column corresponds to a different time step
    piano_roll = pm.get_piano_roll()

    # Normalize the values to 0 or 1
    piano_roll = np.where(piano_roll > 0, 1, 0)

    return np.transpose(piano_roll)


def write_notes_to_pretty_midi(notes, program_number=0, save_to=None) -> PrettyMIDI:
    """
    Create a PrettyMIDI object and populate it with the given notes, using the specified instrument program name.
    Optionally, save the resulting MIDI data to a file.

    :param notes: A list of note objects containing the properties 'velocity', 'pitch', 'start', and 'end' for each note.
    :param program_number: The index of the instrument program to be used (default is 0 or "Acoustic Grand Piano").
    :param save_to: Optional path to save the MIDI file. If provided, the MIDI data will be written to this file.
    :return: The PrettyMIDI object containing the given notes.
    """
    # Create a PrettyMIDI object
    midi = PrettyMIDI()
    # Create an Instrument instance for the specified instrument program
    instrument = Instrument(program=program_number)
    # Iterate over the notes and add them to the instrument
    for note in notes:
        midi_note = Note(
            velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end
        )
        instrument.notes.append(midi_note)
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    if save_to:
        # Write the MIDI data to the specified file
        midi.write(save_to)

    return midi


def shift_notes_start_to_beginning(input_midi) -> PrettyMIDI:
    """
    Shifts all notes to start from the beginning of the track.

    :param: midi_file_path (str): Path to the MIDI file.

    :returns: pretty_midi.PrettyMIDI: A PrettyMIDI object.
    """

    # Check the type of the input
    if isinstance(input_midi, str):
        # If it's a string (file path), load the MIDI file into a PrettyMIDI object
        pm = PrettyMIDI(input_midi)

    elif isinstance(input_midi, PrettyMIDI):
        # If it's a PrettyMIDI object, use it directly
        pm = input_midi
    else:
        # If it's neither, raise an error
        raise TypeError(
            "Input must be either a path to a MIDI file (str) or a PrettyMIDI object"
        )

    # Iterate over all instruments in the MIDI data
    for instrument in pm.instruments:
        if len(instrument.notes) > 1:
            # Sort the notes by their start time
            instrument.notes.sort(key=lambda note: note.start)

            # Get the start time of the first note
            first_note_start = instrument.notes[0].start

            # Iterate over all notes in the instrument
            for i in range(len(instrument.notes) - 1):
                # Shift the start and end times of the note
                instrument.notes[i].start -= first_note_start
                instrument.notes[i].end -= first_note_start

            # Shift the start and end times of the last note
            instrument.notes[-1].start -= first_note_start
            instrument.notes[-1].end -= first_note_start

    return pm


def trim_midi(
    input_midi: PrettyMIDI, start: float, end: float, strict: bool = True
) -> PrettyMIDI:
    """
    Trims midi file using start and end time.

    :param mid (PrettyMidi): input midi file
    :param start (float): start time
    :param end (float): end time
    :param strict (bool, optional):
            If false, includes notes that starts earlier than start time,
            and ends later than start time. Or ends later than end time,
            but starts earlier than end time. The start and end times
            are readjusted so they fit into the given boundaries.
            Defaults to True.

    :return (PrettyMidi): Trimmed output MIDI.
    """
    eps = 1e-3
    midi = deepcopy(input_midi)
    for instrument in midi.instruments:
        if strict:
            instrument.notes = [
                note
                for note in instrument.notes
                if note.start >= start and note.end <= end
            ]
        else:
            instrument.notes = [
                note
                for note in instrument.notes
                if note.end > start + eps and note.start < end - eps
            ]

        for note in instrument.notes:
            if not strict:
                # readjustment
                note.start = max(start, note.start)
                note.end = min(end, note.end)
            # Make the excerpt start at time zero
            note.start -= start
            note.end -= start
    # Filter out empty tracks
    midi.instruments = [ins for ins in midi.instruments if ins.notes]
    return midi


def select_random_scale():
    """
    Selects a random scale from predefined scales.

    Returns:
        list: A list of MIDI note values representing the selected scale.
    """
    scales = [
        [60, 62, 64, 65, 67, 69, 71, 72],  # C major
        [60, 62, 63, 65, 67, 69, 70, 72],  # C natural minor
        [60, 62, 63, 66, 67, 69, 71, 72],  # C harmonic minor
        [60, 62, 63, 65, 67, 69, 71, 72],  # C melodic minor (ascending)
        [60, 62, 64, 67, 69, 72],  # C major pentatonic
        [60, 63, 65, 68, 70, 72],  # C minor pentatonic
        [60, 63, 65, 66, 67, 70, 72],  # C blues scale
        [60, 62, 64, 66, 68, 70, 72],  # C whole tone scale
        [60, 62, 63, 65, 66, 68, 69, 71, 72],  # C diminished scale
    ]
    return random.choice(scales)


def create_midi_melody_example(file_name=None):
    """
    Creates a simple 8-bar melody in the key of C major and saves it as a MIDI file.

    Parameters:
        file_name (str): The name of the MIDI file to be saved (default is 'simple_melody.mid').

    Returns:
        None
    """

    # Create a PrettyMIDI object
    midi = PrettyMIDI()

    # Create a piano instrument
    piano = Instrument(program=0)

    # Define the melody and rhythm
    melody = [
        (4, 0),
        (4, 2),
        (4, 4),
        (4, 2),
        (8, 5),
        (8, 2),
        (8, 1),
        (8, 0),
        (2, 4),
        (4, 6),
        (4, 4),
        (4, 5),
        (4, 1),
        (8, 4),
        (8, 5),
        (8, 2),
        (8, 1),
        (2, 0),
    ] * 2

    # Select a random scale
    scale = select_random_scale()

    # Initialize the start time
    time = 0

    # Iterate through the melody, adding notes to the piano instrument
    for duration, degree in melody:
        # Make sure the degree is within the bounds of the scale
        if degree < len(scale):
            note_duration = 4 / duration
            note_start = time
            note_end = time + note_duration
            note_pitch = scale[degree]
            note_velocity = 100
            piano.notes.append(
                Note(
                    velocity=note_velocity,
                    pitch=note_pitch,
                    start=note_start,
                    end=note_end,
                )
            )
            time += note_duration

    # Add the piano instrument to the PrettyMIDI object
    midi.instruments.append(piano)

    if file_name:
        # Write the PrettyMIDI object to a MIDI file
        midi.write(file_name)
        logger.info("MIDI file created successfully at {}!".format(file_name))
    else:
        logger.info("MIDI file created successfully!")

    return midi
