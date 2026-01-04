"""Agent utilities for rendering and visualizing sequences."""

# Methods to render/visualize seqeunces
import mido
import pretty_midi
from mido import Message, MidiFile, MidiTrack

from src.utils.logging.logging_manager import get_logger

logger = get_logger("agent_utils")


def render_midi(sequence, filename=None, tempo=120):
    """
    Renders a sequence of notes as a MIDI file.

    Args:
        sequence (list): A list of integers where each integer represents a note.
                         The integer 0 corresponds to C4 and 7 corresponds to C5.
        filename (str): The name of the MIDI file to save the output to.
        tempo (int): The tempo of the piece in beats per minute (BPM).

    Returns:
        None: The function creates a MIDI file but does not return any value.
    """
    # Define a dictionary to map the sequence numbers to MIDI note numbers
    # MIDI note number 60 is C4. The mapping is based on the chromatic scale.
    note_mapping = {i: 60 + i for i in range(8)}

    # Create a new MIDI file and track
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Set the tempo (in microseconds per beat)
    microseconds_per_beat = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_beat))

    # Add a note on and note off message for each note in the sequence
    for note in sequence:
        # Convert the sequence number to a MIDI note number
        midi_note = note_mapping[note]
        # Note on message
        track.append(Message("note_on", note=midi_note, velocity=64, time=0))
        # Note off message after one beat (480 ticks in a default MIDI file)
        track.append(Message("note_off", note=midi_note, velocity=64, time=480))

    if filename is not None:
        # Save the MIDI file
        logger.info("File created!")
        midi_file.save(filename)
    else:
        return midi_file


def print_sequence_matrix(sequence, note_range=7):
    """
    Prints the generated sequence in a matrix format using pretty_midi for note names.
    Uses "-" for notes that are not present and separates each step with "|".
    This version is adapted for the C major scale.

    Args:
        sequence (list): A list of integers where each integer represents a note in the C major scale.
                         The integer 0 corresponds to C4, 1 corresponds to D4, and so on up to 6 for B4.
        note_range (int): The range of notes to display in the matrix (y-axis), default is 7 for C major scale.

    Returns:
        None: The function prints the sequence matrix to the output.
    """
    # C major scale MIDI note numbers for one octave starting from C4 (MIDI note number 60)
    c_major_scale_notes = [60, 62, 64, 65, 67, 69, 71]

    # Initialize an empty matrix with the given note range and sequence length
    matrix = [["-" for _ in range(len(sequence))] for _ in range(note_range)]

    # Fill the matrix with 'X' where notes are played
    for step, note in enumerate(sequence):
        if 0 <= note < note_range:
            matrix[note][step] = "X"

    # Print the matrix
    logger.info("Sequence Matrix:")
    logger.info(
        "  " + " | ".join(str(i + 1) for i in range(len(sequence)))
    )  # Print column headers (steps)
    logger.info(
        "+" + "+".join(["---" for _ in range(len(sequence))])
    )  # Print separator line
    for note in range(
        note_range - 1, -1, -1
    ):  # Print rows in reverse order (highest note first)
        note_number = c_major_scale_notes[
            note
        ]  # Get the MIDI note number for the C major scale
        note_name = pretty_midi.note_number_to_name(
            note_number
        )  # Get the note name (e.g., C4, D4, etc.)
        row = " | ".join(matrix[note])
        logger.info(
            "| {} |: {}".format(row, note_name)
        )  # Print the row with the note name
    logger.info(
        "+" + "+".join(["---" for _ in range(len(sequence))])
    )  # Print separator line
