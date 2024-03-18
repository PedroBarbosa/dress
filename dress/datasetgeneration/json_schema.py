schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "generate": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean"},
                "disable_gpu": {"type": "boolean"},
                "verbosity": {"type": "integer"},
                "outdir": {"type": "string"},
                "outbasename": {"type": ["null", "string"]},
                "seed": {"type": "integer"},
                "model": {"type": "string"},
                "model_scoring_metric": {"type": "string"},
                "pangolin_mode": {"type": "string"},
                "pangolin_tissue": {"type": ["null", "string"]},
                "fitness": {
                    "type": "object",
                    "properties": {
                        "minimize_fitness": {"type": "boolean"},
                        "fitness_function": {"type": "string"},
                        "fitness_threshold": {"type": "number"},
                    },
                },
                "archive": {
                    "type": "object",
                    "properties": {
                        "archive_size": {"type": "integer"},
                        "archive_diversity_metric": {"type": "string"},
                        "prune_archive_individuals": {"type": "boolean"},
                        "prune_at_generations": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                    },
                },
                "population": {
                    "type": "object",
                    "properties": {"population_size": {"type": "integer"}},
                },
                "individual": {
                    "type": "object",
                    "properties": {"representation": {"type": "string"}},
                },
                "selection": {
                    "type": "object",
                    "properties": {
                        "selection_method": {"type": "string"},
                        "tournament_size": {"type": "integer"},
                        "custom_mutation_operator": {"type": "boolean"},
                        "custom_mutation_operator_weight": {"type": "number"},
                        "mutation_probability": {"type": "number"},
                        "crossover_probability": {"type": "number"},
                        "operators_weight": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "elitism_weight": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "novelty_weight": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "update_weights_at_generation": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                    },
                },
                "stopping": {
                    "type": "object",
                    "properties": {
                        "stopping_criterium": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "stop_at_value": {
                            "type": "array",
                            "items": {"type": ["integer", "number", "string"]},
                        },
                        "stop_when_all": {"type": "boolean"},
                    },
                },
                "tracking_evolution": {
                    "type": "object",
                    "properties": {
                        "disable_tracking": {"type": "boolean"},
                        "track_full_population": {"type": "boolean"},
                        "track_full_archive": {"type": "boolean"},
                    },
                },
                "grammar": {
                    "type": "object",
                    "properties": {
                        "which_grammar": {"type": "string"},
                        "max_diff_units": {"type": "integer"},
                        "snv_weight": {"type": "number"},
                        "insertion_weight": {"type": "number"},
                        "deletion_weight": {"type": "number"},
                        "acceptor_untouched_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "donor_untouched_range": {
                            "type": "array", 
                            "items": {"type": "integer"},
                        },
                        "untouched_regions": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "max_insertion_size": {"type": "integer"},
                        "max_deletions_size": {"type": "integer"},
                        "motif_db": {"type": "string"},
                        "motif_search": {"type": "string"},
                        "subset_rbps": {"type": "string"},
                        "min_nucleotide_probability": {"type": "number"},
                        "min_motif_length": {"type": "integer"},
                        "pvalue_threshold": {"type": "number"},
                    },
                },
            },
            "required": [
                "dry_run",
                "seed",
                "fitness",
                "archive",
                "population",
                "individual",
                "selection",
                "stopping",
                "tracking_evolution",
                "grammar",
            ],
        }
    },
}


def flatten_dict(d):
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)
