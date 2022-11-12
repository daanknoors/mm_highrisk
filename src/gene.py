"""Gene specific functions"""
import pandas as pd
from Bio import Entrez

from src.preprocess import load_expression_data
from src import config


def create_gene_description_csv(entrez_email):
    """Create dataframe of gene descriptions"""
    df_expression_raw = load_expression_data()

    # get all Entrez ids in expression database
    entrez_ids = list(df_expression_raw.iloc[:, 0].astype(str))

    # get descriptions of each gene
    dict_genes = get_gene_descriptions(entrez_email, entrez_ids)

    # transform to dataframe and save as csv
    df_gene = pd.DataFrame(dict_genes.items(), columns=['EntrezID', 'Description'])
    df_gene.to_csv(config.PATH_DATA / 'gene_descriptions.csv', index=False)
    print(f"Saved gene descriptions at: {config.PATH_MODEL /'gene_descriptions.csv'}")
    return df_gene


def get_single_gene_description(entrez_email, entrez_id, description='full'):
    """Get single description of one gene"""
    Entrez.email = entrez_email

    # convert to list if string
    if isinstance(entrez_id, str):
        entrez_id = [entrez_id]

    # fetch data on entrez ids
    handle_fetch = Entrez.efetch(db="gene", id=entrez_id, retmode='xml')

    # create generator of parsed XML
    results = Entrez.parse(handle_fetch)

    entrez_descriptions = {}
    # retrieve the description of the gene and save to dict
    for i, r in zip(entrez_id, results):
        if description == 'full':
            entrez_descriptions[i] = r
        else:
            entrez_descriptions[i] = r['Entrezgene_gene']['Gene-ref']['Gene-ref_locus']
    handle_fetch.close()
    return entrez_descriptions

def get_gene_descriptions(entrez_email, entrez_ids):
    """Get all gene descriptions. Requires you to specify an Entrez email address"""
    Entrez.email = entrez_email

    # create string of entrez ids separated by comma's
    entrez_ids_string = ",".join(entrez_ids)

    # create connection and specify list of entrez id's to request
    handle_post = Entrez.epost(db="gene", id=entrez_ids_string)
    search_results = Entrez.read(handle_post)
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    print(f'web_env: {webenv}')
    print(f'query_key: {query_key}')

    # fetch data on entrez ids
    handle_fetch = Entrez.efetch(db="gene", retmode='xml', WebEnv=webenv, query_key=query_key)

    # create generator of parsed XML
    results = Entrez.parse(handle_fetch)

    entrez_descriptions = {}

    # retrieve the description of the gene and save to dict
    for i, r in zip(entrez_ids, results):
        entrez_descriptions[i] = r['Entrezgene_gene']['Gene-ref']['Gene-ref_locus']

    handle_fetch.close()
    handle_post.close()
    return entrez_descriptions

