"""Inference and Human-in-the-Loop components.

This package provides:
- InteractiveInferenceSession: Human-in-the-loop feedback collection
- SequenceAnalyzer: Sequence analysis and visualization
- SequenceProfileAnalyzer: Profile-enriched sequence analysis
- BenchmarkRunner: Comparative benchmark evaluation
- PaperResultsCompiler: Publication-ready results generation
- ClusterProfileLoader: Cluster profile metadata loading
- PreferenceScenario: User preference scenario definitions
- PreferenceFeedbackSimulator: Simulated user feedback
- PreferenceGuidedSession: Preference-guided simulation session
- SimulationRunner: Batch simulation execution
- SimulationVisualizer: Publication figure generation
"""

from src.inference.interactive_session import (
    InteractiveInferenceSession,
    InteractiveResult,
    MultidimensionalFeedback,
    MultidimensionalFeedbackCollector,
    MIDIPlayback,
    create_interactive_session,
)

from src.inference.sequence_analysis import (
    SequenceAnalyzer,
    DiversityMetrics,
    TransitionMetrics,
    analyze_sequences,
)

from src.inference.sequence_profile_analysis import (
    SequenceProfileAnalyzer,
    ProfileEnrichedMetrics,
    ArousalMetrics,
    InstrumentMetrics,
    RoleMetrics,
    PreferenceMetrics,
    GenreMetrics,
    MusicalFeatureStats,
    analyze_sequences_with_profiles,
)

from src.inference.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    StatisticalComparison,
    load_benchmark_results,
)

from src.inference.paper_results import (
    LaTeXTableGenerator,
    FigureGenerator,
    PaperResultsCompiler,
    TableConfig,
    generate_paper_results,
)

from src.inference.cluster_profiles import (
    ClusterProfile,
    ClusterProfileLoader,
    load_default_profiles,
)

from src.inference.preference_simulation import (
    PreferenceScenario,
    PreferenceFeedbackSimulator,
    SimulatedFeedback,
    get_predefined_scenarios,
    create_scenario_from_profiles,
)

from src.inference.preference_guided_session import (
    PreferenceGuidedSession,
    SimulationResult,
    SimulationMetrics,
)

from src.inference.simulation_runner import (
    SimulationRunner,
)

from src.inference.simulation_visualizer import (
    SimulationVisualizer,
)

__all__ = [
    # Interactive Session
    "InteractiveInferenceSession",
    "InteractiveResult",
    "MultidimensionalFeedback",
    "MultidimensionalFeedbackCollector",
    "MIDIPlayback",
    "create_interactive_session",
    # Sequence Analysis
    "SequenceAnalyzer",
    "DiversityMetrics",
    "TransitionMetrics",
    "analyze_sequences",
    # Sequence Profile Analysis
    "SequenceProfileAnalyzer",
    "ProfileEnrichedMetrics",
    "ArousalMetrics",
    "InstrumentMetrics",
    "RoleMetrics",
    "PreferenceMetrics",
    "GenreMetrics",
    "MusicalFeatureStats",
    "analyze_sequences_with_profiles",
    # Benchmark
    "BenchmarkResult",
    "BenchmarkRunner",
    "StatisticalComparison",
    "load_benchmark_results",
    # Paper Results
    "LaTeXTableGenerator",
    "FigureGenerator",
    "PaperResultsCompiler",
    "TableConfig",
    "generate_paper_results",
    # Cluster Profiles
    "ClusterProfile",
    "ClusterProfileLoader",
    "load_default_profiles",
    # Preference Simulation
    "PreferenceScenario",
    "PreferenceFeedbackSimulator",
    "SimulatedFeedback",
    "get_predefined_scenarios",
    "create_scenario_from_profiles",
    # Preference Guided Session
    "PreferenceGuidedSession",
    "SimulationResult",
    "SimulationMetrics",
    # Simulation Runner
    "SimulationRunner",
    # Simulation Visualizer
    "SimulationVisualizer",
]
