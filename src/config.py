"""Configuration settings"""
import os
from pathlib import Path

# paths
PATH_PROJECT = Path(os.getcwd()).parent
PATH_DATA = PATH_PROJECT / 'data'
PATH_MODEL = PATH_PROJECT / 'models'


# filenames
FILENAME_CLINICAL_DATA = 'sc3_Training_ClinAnnotations.csv'
FILENAME_EXPRESSION_DATA = 'MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv'
FILENAME_DICTIONARY_DATA = 'Harmonized_Clinical_Dictionary.csv'

# config settings
RANDOM_STATE = 42
COlOR_PALETTE = ["#393e46", "#00cde3", "#ff5722", "#d72323"]

# features categories

FEATURES_DEMOGRAPHICS = ['D_Age', 'D_Gender', 'D_OS', 'D_OS_FLAG', 'D_ISS', 'D_PFS', 'D_PFS_FLAG']
FEATURES_CYTO = ['CYTO_predicted_feature_01', 'CYTO_predicted_feature_02', 'CYTO_predicted_feature_03',
                 'CYTO_predicted_feature_04', 'CYTO_predicted_feature_05', 'CYTO_predicted_feature_06',
                 'CYTO_predicted_feature_07', 'CYTO_predicted_feature_08', 'CYTO_predicted_feature_09',
                 'CYTO_predicted_feature_10', 'CYTO_predicted_feature_11', 'CYTO_predicted_feature_12',
                 'CYTO_predicted_feature_13', 'CYTO_predicted_feature_14', 'CYTO_predicted_feature_15',
                 'CYTO_predicted_feature_16', 'CYTO_predicted_feature_17', 'CYTO_predicted_feature_18']

FEATURES_NOMINAL = ['D_Gender']
FEATURES_ORDINAL = ['D_Age', 'D_ISS']

FEATURES_DROP = ['SampleID', 'Study', 'Patient', 'D_OS', 'D_PFS', 'D_OS_FLAG', 'D_PFS_FLAG',
                 'PatientType', 'MA_probeLevelExpFile',
                 'MA_probeLevelExpFileSamplId', 'MA_geneLevelExpFile',
                 'MA_geneLevelExpFileSamplId', 'RNASeq_transLevelExpFile',
                 'RNASeq_transLevelExpFileSamplId', 'RNASeq_geneLevelExpFile',
                 'RNASeq_geneLevelExpFileSamplId', 'WES_mutationFileMutect',
                 'WES_mutationFileStrelkaIndel', 'WES_mutationFileStrelkaSNV',
                 'RNASeq_mutationFileMutect', 'RNASeq_mutationFileStrelkaIndel',
                 'RNASeq_mutationFileStrelkaSNV', 'RNASeq_FusionFile']

# target columns
TARGET = 'HR_FLAG'
TARGET_EXTRA = ['D_OS', 'D_PFS']  # correlate with TARGET




