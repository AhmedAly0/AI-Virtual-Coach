"""
Utility functions for GEI exercise recognition project.
"""

from .io_utils import (
    setup_results_folder, 
    setup_results_folder_for_backbone, 
    set_global_seed,
    display_results_structure,
    load_config
)
from .metrics import (
    save_experiment_summary,
    save_backbone_config,
    get_model_parameters,
    get_all_model_parameters,
    load_backbone_results_with_config
)
from .visualization import (
    save_training_curves,
    save_training_curves_train_only,
    save_confusion_matrix,
    create_comprehensive_comparison,
    generate_statistical_comparison,
    plot_backbone_comparison,
    plot_aggregated_confusion_matrix
)

__all__ = [
    'setup_results_folder',
    'setup_results_folder_for_backbone',
    'set_global_seed',
    'display_results_structure',
    'load_config',
    'save_experiment_summary',
    'save_backbone_config',
    'get_model_parameters',
    'get_all_model_parameters',
    'load_backbone_results_with_config',
    'save_training_curves',
    'save_training_curves_train_only',
    'save_confusion_matrix',
    'create_comprehensive_comparison',
    'generate_statistical_comparison',
    'plot_backbone_comparison',
    'plot_aggregated_confusion_matrix',
]
