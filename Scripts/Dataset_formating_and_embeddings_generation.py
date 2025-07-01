import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import csv
import re

data_dir = "Data"
filtered_df=pd.read_csv('Filtered_OFF.csv',sep='\t')
filtered_df["code"] = filtered_df["code"].astype(str)
non_nutrient_columns=['code', 'product_name', 'nova_group', 'ingredients_text', 'additives_n',
       'additives', 'standard_ingredients', 'ingredient_count']

nutrient_columns = [col for col in filtered_df.columns if col not in non_nutrient_columns]

# Function to clean ingredient text to calculate the number of ingredients
def clean_ingredients(text):
    no_paren = re.sub(r"\([^)]*\)", "", text)
    no_brack = re.sub(r"\[[^]]*\]", "", no_paren)
    parts = [tok.strip() for tok in no_brack.split(',') if tok.strip()]
    return len(parts)

# Compute number of ingredients and add as a new column
filtered_df['num_ingredients'] = filtered_df['ingredients_text'].astype(str).apply(clean_ingredients)

# Function to format nutrient names
def format_nutrient_name(nutrient):
    return nutrient.replace("_", " ").replace("-", " ").strip()

# Function to create sentences
def create_sentence(row):
    product_name = str(row["product_name"]).strip()
    ingredients = str(row["ingredients_text"]).strip()
    nutrient_list = []  
    for col in nutrient_columns:
        val = row[col]
        if pd.notna(val):  # Only include non-null values
            nutrient_name = format_nutrient_name(col.split("_100g")[0])  # Remove '_100g' and format name
            nutrient_list.append(f"{val} grams of {nutrient_name}")   
    nutrients_str = ", ".join(nutrient_list)
    return f"{product_name} has the ingredients: {ingredients}, and the nutrients: {nutrients_str}." if nutrients_str else f"{product_name} has the ingredients: {ingredients}."

# Create sentences
filtered_df["sentence"] = filtered_df.apply(create_sentence, axis=1)
filtered_df.to_csv(
    data_dir+"Filtered_OFF_with_sentences.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_ALL,
    encoding='utf-8'
)


# ──────────────────────────────────────────────────────────────────────────────
# BERT embeddings
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = pd.read_csv(
    data_dir+"Filtered_OFF_with_sentences.csv',
    sep='\t',
    quoting=csv.QUOTE_ALL,
    encoding='utf-8'
)

filtered_df["code"] = filtered_df["code"].astype(str)

# Load BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to get BERT embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Extract [CLS] token

embeddings = [get_bert_embedding(sent) for sent in filtered_df["sentence"]]

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Insert 'code' and 'NOVA_class' columns
embeddings_df.insert(0, "nova_group", filtered_df["nova_group"].astype(int).values)
embeddings_df.insert(0, "code", filtered_df["code"].values)

# Rename embedding columns to 0...767
embedding_dim = embeddings_df.shape[1] - 2
embeddings_df.columns = ["code", "nova_group"] + [str(i) for i in range(embedding_dim)]

# Save as TSV file
output_file = data_dir+"bert_embeddings.tsv"
embeddings_df.to_csv(output_file, sep="\t", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# BioBERT embeddings
# ──────────────────────────────────────────────────────────────────────────────

# Load BioBERT
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Batch embedding function
def get_batched_embeddings(sentences, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="BioBERT Batch Embedding"):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] tokens
        all_embeddings.append(cls_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()

# Compute batched embeddings
embeddings = get_batched_embeddings(sentences, batch_size=32)

# Save to DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, "nova_group", filtered_df["nova_group"].astype(int).values)
embeddings_df.insert(0, "code", filtered_df["code"].values)

# Rename columns to match 0...767
embedding_dim = embeddings_df.shape[1] - 2
embeddings_df.columns = ["code", "nova_group"] + [str(i) for i in range(embedding_dim)]

# Save as TSV
output_file = data_dir+"bio_bert_embeddings.tsv"
embeddings_df.to_csv(output_file, sep="\t", index=False)
