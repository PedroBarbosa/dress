schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "evaluate": {
            "type": "object",
            "properties": {
                "outdir": {"type": "string"},
                "outbasename": {"type": "string"},
                "verbosity": {"type": "integer"},
                "input": {
                    "type": "object",
                    "another_dataset": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "input_seq": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "string"},
                        ]
                    },
                    "another_input_seq": {
                        "anyOf": [
                            {"type": "null"},
                            {"type": "string"},
                        ]
                    },
                    "groups": {"type": ["null", "string"]},
                    "list": {"type": "boolean"},
                },
                "evo_alg":{
                    "type": "object",
                    "properties":{
                        "seed": {"type": "integer"},
                        "minimize_fitness": {"type": "boolean"},
                        "fitness_function": {"type": "string"},
                        "population_size": {"type": "integer"},
                        "stopping_criterium": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "stop_at_value": {
                            "type": "array",
                            "items": {"type": ["integer", "number", "string"]},
                        },
                        "stop_when_all": {"type": "boolean"},
                        "selection_method": {"type": "string"},
                        "tournament_size": {"type": "integer"},
                        "operators_weight": {"type": "number"},
                        "elitism_weight": {"type": "number"},
                        "novelty_weight": {"type": "number"},
                        "mutation_probability": {"type": "number"},
                        "crossover_probability": {"type": "number"},
                        "simplify_explanation": {"type": "boolean"},
                        "simplify_at_generations": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                    }
                },
                "motifs": {
                    "type": "object",
                    "properties": {
                        "motif_db": {"type": "string"},
                        "motif_search": {"type": "string"},
                        "subset_rbps": {"type": "string"},
                        "min_nucleotide_probability": {"type": "number"},
                        "min_motif_length": {"type": "integer"},
                        "pvalue_threshold": {"type": "number"},
                        "qvalue_threshold": {"type": "number"},
                        "pssm_threshold": {"type": "integer"},
                        "motif_results":  {"type": ["null", "string"]},
                    },
                },
            },
            "required": [
                "outdir",
                "outbasename",
                "verbosity",
                "evo_alg",
                "motifs",
            ],
        }
    },
}
