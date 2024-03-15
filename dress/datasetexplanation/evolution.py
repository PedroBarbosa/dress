import sqlite3
from typing import Union
from pypika import Query, Table, Case, Field
from pypika import functions as fn
from dress.datasetgeneration.dataset import Dataset, PairedDataset
from .grammars.motif_based_grammar import create_grammar
from .config_evolution import configureEvolution

def split_dataset(data, test_size=0.2):
    """
    Splits the dataset into a training and a test set

    Args:
        data (pd.DataFrame): Data to be split
        test_size (float): Size of the test set

    Returns:
        tuple: Training and test sets
    """
    from sklearn.model_selection import train_test_split
    train_X, test_X = train_test_split(data, test_size=test_size)
    return train_X, test_X

def extract_rbp_list(db: sqlite3.Connection) -> list:
    """
    Returns the list of RBPs with any hit in the database
    
    Args:
        db (sqlite3.Connection): Cursor object to execute SQL queries
        
    Returns:
        list: List of unique RBP names present in the database
    """
    motifs = Table("motifs")
    q = Query.from_(motifs).select("rbp_name").distinct()
    rbp_names = db.execute(str(q)).fetchall()
    return [item for sublist in rbp_names for item in sublist]

def do_evolution(
    dataset_obj: Union[Dataset, PairedDataset],
    db: sqlite3.Connection,
    **kwargs,
):
    """
    Evolution of an explanation using GeneticEngine

    Args:
        dataset_obj (Dataset): Dataset to explain
        db (sqlite3.Connection): Database with the motifs
    """
    #train_X, test_X= split_dataset(dataset_obj.data, test_size=0.2)

    grammar = create_grammar(
        rbp_list=extract_rbp_list(db),
        max_n_rules=10,
        motif_presence_weight=0.2,
        motif_co_occurrence_weight=0.2,
        motif_inter_distance_weight=0.2,
        motif_ss_distance_weight=0.2,
    )

    alg = configureEvolution(
        dataset_obj=dataset_obj,
        grammar=grammar,
        **kwargs,
    )

    kwargs['logger'].info("Evolution started")
    alg.evolve()