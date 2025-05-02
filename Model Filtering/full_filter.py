# %%
import os
import dask
import dask.dataframe as dd
import pandas as pd
import nltk
import torch
import textwrap
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from dask.diagnostics import ProgressBar

# ⚙️ Reduce threading to avoid kernel crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ⚙️ Dask thread-safe mode to avoid torch fork() crashes
dask.config.set(scheduler='threads')

# %%
nltk.download('punkt')

# Load sentence embedding model once globally
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define diabetes-related reference terms
diabetes_terms = [
    "diabetes", "diabetic", "DM1", "DM Type 1", "DM Type I", "type 1 diabetes", "type 2 diabetes",
    "diabetes diagnosis", "DKA", "ketoacidosis", "DKA history", "DKA episode", "DKA resolved", "DKA management",
    "hyperglycemia", "hypoglycemia", "hyperglycemia treatment",
    "polyuria", "polydipsia", "polyphagia", "HPI", "blood sugar", "blood sugar control",
    "glucose", "glucose-", "FS=", "FS:", "FS-", "FS(", "HbA1C", "a1c", "A1C test results",
    "ketone", "anion gap", "bicarbonate", "blood glucose", "blood sugars",
    "insulin", "insulin therapy", "Lantus", "Novalog", "Humalog",
    "sliding scale", "sliding scale insulin", "subcutaneously", "insulin drip", "insulin gtt", "insulin sliding scale", "amp D50",
    "carbohydrate", "diabetic diet", "glucose control"
]

# Encode reference embeddings once globally
ref_embeds = model.encode(diabetes_terms, convert_to_tensor=True)

# Native replacement for more_itertools.chunked
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# %%
# Load glucose-insulin dataset
df_glucose = pd.read_csv(
    "curated-data-for-describing-blood-glucose-management-in-the-intensive-care-unit-1.0.1/Datasets/glucose_insulin_ICU.csv",
    dtype={'SUBJECT_ID': str, 'HADM_ID': str}
)
glucose_ids = set(df_glucose["SUBJECT_ID"])

# Clean malformed NOTEEVENTS lines using Pandas
df_cleaned = pd.read_csv(
    "mimic-iii-clinical-database-1.4/NOTEEVENTS.csv",
    dtype=str,
    on_bad_lines='skip',
    engine='python'
)
df_cleaned.to_csv("clean_NOTEEVENTS.csv", index=False)

# Load with Dask
ddf = dd.read_csv(
    "clean_NOTEEVENTS.csv",
    dtype=str,
    blocksize="100MB",
    assume_missing=True,
    low_memory=False
)
ddf.columns = ddf.columns.str.strip('"')  # Remove quotes if present

print("Columns loaded:", ddf.columns)

# Filter to non-null TEXT and valid SUBJECT_IDs
ddf = ddf.dropna(subset=["TEXT"])
ddf = ddf[ddf["SUBJECT_ID"].isin(glucose_ids)]

# Persist to break compute graph
ddf = ddf.persist()

# %%
# Define filtering function
import traceback

def extract_matches(note_text):
    try:
        sentences = sent_tokenize(note_text)
        candidates = [s for s in sentences if any(term.lower() in s.lower() for term in diabetes_terms)]
        if not candidates:
            return []

        matches = []
        for chunk in chunked(candidates, 16):  # conservative batch size
            embs = model.encode(chunk, convert_to_tensor=True)
            sims = util.cos_sim(embs, ref_embeds)
            scores = torch.max(sims, dim=1).values
            matches.extend([(s, round(score.item(), 3)) for s, score in zip(chunk, scores) if score > 0.5])

        return matches
    except Exception:
        traceback.print_exc()
        return []

# Dask-safe partition function
def extract_partition(part):
    part["DIABETES_MATCHES"] = part["TEXT"].apply(extract_matches)
    return part

# Apply with map_partitions
ddf = ddf.map_partitions(extract_partition, meta=ddf)

# %%
# Filter rows with at least one match
ddf_filtered = ddf[ddf["DIABETES_MATCHES"].map(lambda x: len(x) > 0, meta=("filter", bool))]

# %%
# Compute and save
with ProgressBar():
    df_result = ddf_filtered[["SUBJECT_ID", "HADM_ID", "DIABETES_MATCHES"]].compute()

df_result.to_json("filtered_diabetes_notes_dask.jsonl", orient="records", lines=True)

# %%
# Preview
if not df_result.empty:
    print("✅ Preview from one filtered note:\n")
    for sent, score in df_result["DIABETES_MATCHES"].iloc[0]:
        print(f"[score={score}] {textwrap.fill(sent, width=100)}")
        print("=" * 120)
else:
    print("⚠️ No diabetes-relevant notes found.")
