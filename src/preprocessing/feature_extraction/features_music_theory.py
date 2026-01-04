"""Music theory-based feature extraction utilities."""

# Inspired by:
# https://github.com/sebasgverde/music-geometry-eval/blob/master/music_geometry_eval/music_geometry_eval.py

import math
from collections import Counter


def calculate_centricity(song, span_size=12):
    """
    Calculate the centricity of a song using a sliding window technique.

    This function computes the centricity by averaging the frequency of the most common note
    or note group within each window across the song.

    The rersult tends to be 1 the more predominant is one of the notes with respect to the others
    and tends to be 0 the less frequent is the most frequent note.

    Parameters:
    - song: A list of MIDI notes (integers) representing a song.
    - span_size: The size of the span (window) to consider for local centricity calculation.

    Returns:
    - float: The average centricity across all windows in the song.
    """

    def window_centricity(window):
        """
        Calculate the centricity of a window (subsong).

        This is done by finding the frequency of the most common note or note group in the window.

        Parameters:
        - window: A list of notes representing a window of the song.

        Returns:
        - float: The frequency of the most common note or note group in the window.
        """
        # Count the occurrences of each note or note group in the window
        note_counts = Counter(window)
        # Find the most common note or note group and its count
        most_common_note, most_common_count = note_counts.most_common(1)[0]
        # Calculate the frequency of the most common note or note group
        frequency = most_common_count / len(window)

        return frequency

    # Initialize a list to hold centricity scores for each window
    window_scores = []

    # Calculate the number of windows to consider
    number_of_windows = abs(len(song) - span_size + 1)
    # Use a sliding window approach to calculate centricity for each window
    for i in range(number_of_windows):
        # Calculate centricity for the current window
        window_scores.append(window_centricity(song[i : i + span_size]))

    # Calculate the average centricity across all windows
    average_centrality = (
        sum(window_scores) / number_of_windows if number_of_windows > 0 else 0
    )

    # Return the average centricity as a float
    return float(average_centrality)


def calculate_weighted_limited_macroharmony(
    song, span_size=12, lower_limit=5, upper_limit=8
):
    """
    Calculate the weighted macroharmony score of a song using a quadratic penalty for deviations
    from the desired range of note diversity.

    The weighted macroharmony score reflects the the diversity of notes within a song or spans of the
    song, with a focus on a specific range of note diversity. The score is highest when the number
    of unique notes falls within the desired range (lower_limit to upper_limit). The score decreases
    quadratically as the number of unique notes moves away from the limits.

     Result is 1 if every span in a melody is within the appropriate limits, or greater than one
     for songs with a large diversity of notes.

    Parameters:
    - song: A list of MIDI notes (integers) representing a song.
    - span_size: The size of the span (window) to consider for local macroharmony calculation.
    - lower_limit: The lower limit of the desired note diversity range.
    - upper_limit: The upper limit of the desired note diversity range.

    Returns:
    - float: The calculated weighted macroharmony score of the song.
    """

    def local_weighted_macrohar(subsong, lower_limit, upper_limit):
        """
        Calculate the local weighted macroharmony score for a subsong using a quadratic penalty.

        Parameters:
        - subsong: A list of notes representing a part of the song.
        - lower_limit: The lower limit of the desired note diversity range.
        - upper_limit: The upper limit of the desired note diversity range.

        Returns:
        - float: The local weighted macroharmony score for the subsong.
        """
        # Count the number of unique notes in the subsong
        number_of_notes = len(set(subsong))
        # Calculate the score based on the number of unique notes
        if lower_limit <= number_of_notes <= upper_limit:
            return 1.0
        elif number_of_notes < lower_limit:
            return math.exp(-((lower_limit - number_of_notes) ** 2))
        else:
            return math.exp(-((number_of_notes - upper_limit) ** 2))

    # Initialize a list to hold scores for each span
    span_scores = []

    # Calculate the number of spans to consider
    number_of_spans = max(1, len(song) - span_size + 1)

    # Use a sliding window approach to calculate the score for each span
    for i in range(number_of_spans):
        # Calculate the score for the current span
        span_scores.append(
            local_weighted_macrohar(song[i : i + span_size], lower_limit, upper_limit)
        )

    # Calculate the average score across all spans
    average_score = sum(span_scores) / number_of_spans

    # Return the average score as a float
    return float(average_score)


def calculate_conjunct_melodic_motion(song):
    """
    Calculate the average melodic motion between consecutive notes in a song using list comprehension.

    This function computes the average 'step size' between consecutive notes in a
    list of pitches, which represents a song, using list comprehension for a more
    concise implementation. The step size is the absolute difference in pitch between
    each pair of consecutive notes. This can be used to measure how 'conjunct' or
    'disjunct' a melody is. A lower average indicates more conjunct motion
    (smaller intervals), while a higher average indicates more disjunct motion
    (larger intervals).
    Result would be 1 if every change is of 1 semitone, tends to be greater than one,
    if there are changes of more than one semitone, and less than 1 for melodies with repeated notes.
    The best value is one because it is a song in the chromatic scale and sounds consonant by defect.

    Parameters:
    song (list of int): A list of MIDI notes (integers) representing a song.

    Returns:
    float: The average melodic motion between consecutive notes.
    """

    # Use list comprehension to calculate the absolute differences between consecutive notes.
    # The zip function is used to create pairs of consecutive notes.
    changes = [
        abs(next_note - current_note) for current_note, next_note in zip(song, song[1:])
    ]

    # Calculate the average change in pitch by summing all the changes
    # and dividing by the number of changes.
    average_motion = sum(changes) / float(len(changes))

    # Return the average melodic motion.
    return average_motion


def get_music_theory_features(notes):
    """
    Extracts a set of music theory features from a sequence of musical notes.

    This function computes various music theory-related features from a given list of notes.
    Each feature captures a different aspect of the musical structure. The features calculated
    are centricity, limited macroharmony, and conjunct melodic motion.

    Parameters:
    notes (list of int): A list of MIDI notes (integers) representing a song.

    Returns:
    dict: A dictionary containing the following key-value pairs:
        - 'centricity': A float representing the centricity of the notes, which is a measure
                        of the prevalence of the most common pitch class in the sequence.
        - 'limited_macroharmony': A float representing the weighted limited macroharmony, which
                                  is a measure of the diversity of pitch classes, weighted by
                                  their frequency of occurrence.
        - 'conjuction_melody_motion': A float representing the average conjunct melodic motion,
                                      which measures the average step size between consecutive
                                      notes, indicating how smooth or disjointed the melody is.

    Note:
    The functions are implemented based on the book by Dimitri Tymoczko, A Geometry of Music
    https://dmitri.mycpanel.princeton.edu/geometry-of-music.html
    """

    cc = calculate_centricity(notes)
    lm = calculate_weighted_limited_macroharmony(notes)
    cmm = calculate_conjunct_melodic_motion(notes)
    gt = cc + lm + cmm

    return {
        "centricity": cc,
        "limited_macroharmony": lm,
        "conjuction_melody_motion": cmm,
        "general_tonality": gt,
    }
