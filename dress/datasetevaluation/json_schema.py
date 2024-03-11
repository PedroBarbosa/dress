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
                "motif_evaluator": {
                    "type": "object",
                    "properties": {
                        "motif_db": {"type": "string"},
                        "motif_search": {"type": "string"},
                        "subset_rbps": {"type": "string"},
                        "skip_raw_motifs_filtering": {"type": "boolean"},
                        "min_nucleotide_probability": {"type": "number"},
                        "min_motif_length": {"type": "integer"},
                        "pvalue_threshold": {"type": "number"},
                        "qvalue_threshold": {"type": "number"},
                        "pssm_threshold": {"type": "integer"},
                        "just_estimate_pssm_threshold": {"type": "boolean"},
                        "motif_counts": {"type": "string"},
                        "motif_enrichment": {"type": "string"},
                    },
                },
            },
            "required": [
                "outdir",
                "outbasename",
                "verbosity",
                "input",
                "motif_evaluator",
            ],
        }
    },
}
