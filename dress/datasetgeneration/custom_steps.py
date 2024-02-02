from copy import deepcopy
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import MutationOperator, Representation

from geneticengine.core.problems import Problem


from geneticengine.core.grammar import Grammar

from dress.datasetgeneration.grammars.with_indels_grammar import DiffUnit


def custom_mutation_operator(
    grammar: Grammar, input_seq: dict, **kwargs
) -> MutationOperator:
    """Creates a custom mutation operator for a GenericMutation step, based on the given grammar.

    Args:
        grammar (Grammar): Grammar that will be used in the evolution
        input_seq (str): Info about the originalsequence
        kwargs (dict): Additional keyword arguments that will be used to create the mutation operator
    Returns:
        GenericMutationStep: A step that applies a mutation operator to the population
    """
    assert "max_deletion_size" in kwargs
    assert "max_insertion_size" in kwargs

    seq = input_seq["seq"]
    seq_size = len(seq)
    root = grammar.starting_symbol
    options = grammar.alternatives[DiffUnit]
    SNV = options[0]
    RandomDeletion = options[1]
    RandomInsertion = options[2]

    class NearbyReplacementOperator(MutationOperator[root]):
        """Custom mutation operator that replaces a diff unit
        of a given type with another diff unit of the same type.
        The replacement occurs at a location very close to the
        original, chosen from a normal distribution centered around
        the original position.

        This operator ensures some constrains for the replacement:
            - For SNVs, the position is different.
            - For deletions, the start position is different if max_deletion_size is 1.
        If max_deletion_size is > 1, the deletion is different.
            - For insertions, the position is different.
        """

        def mutate(
            self,
            genotype: root,  # type: ignore
            problem: Problem,
            evaluator: Evaluator,
            representation: Representation,
            random_source: Source,
            index_in_population: int,
            generation: int,
        ) -> root:  # type: ignore
            nucleotides = ["A", "C", "G", "T"]
            weights = [0.3, 0.2, 0.2, 0.3]

            cpy = deepcopy(genotype)
            el = random_source.randint(0, len(cpy.diffs) - 1)  # type: ignore
            to_be_mutated = cpy.diffs[el]  # type: ignore

            if isinstance(to_be_mutated, SNV):
                pos = int(random_source.normalvariate(to_be_mutated.position, 4))
                nuc = random_source.choice(nucleotides)  # type: ignore

                while (
                    not 0 <= pos < seq_size
                    or pos == to_be_mutated.position
                    or seq[pos] == nuc
                ):
                    pos = int(random_source.normalvariate(to_be_mutated.position, 4))
                    nuc = random_source.choice(nucleotides)  # type: ignore

                to_be_mutated.position = pos
                to_be_mutated.nucleotide = nuc

            elif isinstance(to_be_mutated, RandomDeletion):
                max_size = kwargs["max_deletion_size"]
                if max_size == 1:
                    size = 1
                    pos = int(random_source.normalvariate(to_be_mutated.position, 4))

                    while not 0 <= pos < seq_size or pos == to_be_mutated.position:
                        pos = int(
                            random_source.normalvariate(to_be_mutated.position, 4)
                        )

                else:
                    size = random_source.randint(1, max_size)

                    # Mean is the middle position of the deletion
                    mean_pos = (
                        to_be_mutated.position
                        + to_be_mutated.position
                        + to_be_mutated.size
                        - 1
                    ) // 2
                    pos = int(random_source.normalvariate(mean_pos, 4))

                    while not 0 <= pos < seq_size - size or size == to_be_mutated.size:
                        pos = int(random_source.normalvariate(mean_pos, 4))
                        size = random_source.randint(1, max_size)

                to_be_mutated.position = pos
                to_be_mutated.size = size

            elif isinstance(to_be_mutated, RandomInsertion):
                max_size = kwargs["max_insertion_size"]
                pos = int(random_source.normalvariate(to_be_mutated.position, 4))
                while not 0 <= pos < seq_size or pos == to_be_mutated.position:
                    pos = int(random_source.normalvariate(to_be_mutated.position, 4))

                size = random_source.randint(1, max_size)
                nuc = "".join(
                    random_source.choice_weighted(nucleotides, weights)
                    for _ in range(size)
                )

                to_be_mutated.position = pos
                to_be_mutated.nucleotides = nuc

            return cpy

    return NearbyReplacementOperator()
