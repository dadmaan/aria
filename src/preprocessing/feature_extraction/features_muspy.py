"""Feature extraction utilities using the MusPy library."""

import muspy


def get_midi_metrics_from_muspy(file_path):
    """
    Calculate various metrics for a MIDI file using the MusPy library.

    Parameters:
    - file_path (str): The file path of the MIDI file.

    Returns:
    - dict: A dictionary containing calculated metrics.
    """
    try:
        # Load MIDI file using MusPy
        midi = muspy.read_midi(file_path)

        # Calculate various metrics
        metrics = {
            "pitch_range": muspy.pitch_range(midi),
            "n_pitches_used": muspy.n_pitches_used(midi),
            "n_pitch_classes_used": muspy.n_pitch_classes_used(midi),
            "polyphony": muspy.polyphony(midi),
            # 'polyphony_rate': muspy.polyphony_rate(midi, root, mode),
            "scale_consistency": muspy.scale_consistency(midi),
            "pitch_entropy": muspy.pitch_entropy(midi),
            "pitch_class_entropy": muspy.pitch_class_entropy(midi),
            "empty_beat_rate": muspy.empty_beat_rate(midi),
            "drum_in_pattern_rate_duple": muspy.drum_in_pattern_rate(midi, "duple"),
            "drum_in_pattern_rate_triple": muspy.drum_in_pattern_rate(midi, "triple"),
            "drum_pattern_consistency": muspy.drum_pattern_consistency(midi),
            "groove_consistency": muspy.groove_consistency(midi, midi.resolution),
            "empty_measure_rate": muspy.empty_measure_rate(midi, midi.resolution),
        }

        return metrics

    except Exception as e:
        # Handle exceptions, e.g., file not found or invalid MIDI format
        return {"error": str(e)}
