import argparse
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from simcse import SimCSE
from tqdm import tqdm

def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Using CLS token embedding
    return embeddings

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def load_suffix_scores(completions_file):
    completions = {}
    with open(completions_file, 'r') as f:
        for line in f:
            prefix, suffix_scores_str = line.strip().split('\t')
            suffix_scores = eval(suffix_scores_str)
            completions[prefix] = suffix_scores
    return completions

# def main(context_suffix_file, completions_file, eval_file, doc_file, bert=1):
#     if bert == -1:
#         with open(doc_file, "r") as f:
#             text = f.read()
#         vectorizer = TfidfVectorizer() 
#         vectorizer.fit([text])
        
#         completions = load_suffix_scores(completions_file)
#         with open(context_suffix_file, 'r') as f:
#             for line in tqdm(f):
#                 line = line.replace('<eou>', '[EOU]')
#                 last_eou_index = line.rfind('[EOU]')  # Find the index of last <eou>
#                 context = line[:last_eou_index]  # Context is everything before the last <eou>
#                 prefix_suffix = line[last_eou_index + 5:].strip()
#                 prefix, suffix = prefix_suffix.split('\t')
#                 prefix = prefix.strip()
                
#                 max_score = -np.inf
#                 best_suffix = ""
#                 prediction = ""
#                 if prefix in completions:
#                     # print("context:", context)
#                     # print("prefix:", prefix)
#                     # print("gt suffix:", suffix)
                    
#                     for predicted_suffix, score_str in completions[prefix]:
#                         # print("PREDICTED SUFFIX: ", predicted_suffix)
#                         predicted_suffix, score = predicted_suffix.strip(), float(score_str.split(":")[1])
#                         tfidf_matrix = vectorizer.transform([context, predicted_suffix])
#                         cosine_sim = 1 - cosine(tfidf_matrix[0].toarray()[0], tfidf_matrix[1].toarray()[0])
                          
#                         scaled_score = score * cosine_sim
#                         if scaled_score > max_score:
#                             max_score = scaled_score
#                             best_suffix = predicted_suffix
                            
#                         # print(f"For prefix '{prefix}', predicted suffix '{predicted_suffix}' with score {score:.4f} and cosine sim {cosine_sim:.4f}")
#                     if(best_suffix is not None):
#                         prediction = best_suffix[len(prefix):]
#                     # suffix = suffix[:-2]
#                     if max_score == -np.inf:
#                         max_score = 0.0
#                     with open(eval_file, "a") as f:
#                         f.write(f'{prefix}\t{suffix}\t{prediction}\t{max_score}\t1\n')          
                    
#     else:
#         # Load BERT model and tokenizer
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         model = BertModel.from_pretrained("bert-base-uncased")
#         model.eval()
        
#         # Load simCSE model
#         model2 = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

#         # Load completions file
#         completions = load_suffix_scores(completions_file)
#         # Process context and suffixes
#         with open(context_suffix_file, 'r') as f:
#             for line in f:
#                 # print(line)
#                 line = line.replace('<eou>', '[EOU]')
#                 last_eou_index = line.rfind('[EOU]')  # Find the index of last <eou>
#                 context = line[:last_eou_index]  # Context is everything before the last <eou>
#                 prefix_suffix = line[last_eou_index + 5:].strip()
                
#                 prefix, suffix = prefix_suffix.split('\t')
#                 prefix = prefix.strip()
#                 # if prefix in completions:
#                 # print("context:", context)
#                 # print("prefix:", prefix)
#                 # print("gt suffix:", suffix)
#                 if(bert):
#                     context_embedding = get_bert_embeddings(context, model, tokenizer)
#                     prefix_embedding = get_bert_embeddings(prefix, model, tokenizer)
#                 max_score = -np.inf
#                 best_suffix = None
                
#                 for predicted_suffix, score_str in completions[prefix]:
#                     predicted_suffix, score = predicted_suffix.strip(), float(score_str.split(":")[1])
#                     if(bert):
#                         suffix_embedding = get_bert_embeddings(predicted_suffix, model, tokenizer)
#                         cosine_sim = calculate_cosine_similarity(context_embedding, suffix_embedding) - calculate_cosine_similarity(context_embedding, prefix_embedding)
#                     else:
#                         cosine_sim = model2.similarity(context, predicted_suffix) - model2.similarity(context, prefix)
#                     # print(f"For prefix '{prefix}', predicted suffix '{predicted_suffix}' with score {score:.4f} and cosine sim {cosine_sim:.4f}")
#                     scaled_score = score * cosine_sim
#                     if scaled_score > max_score:
#                         max_score = scaled_score
#                         best_suffix = predicted_suffix
                
#                 if(best_suffix is not None):
#                     prediction = best_suffix[len(prefix):]
#                 # suffix = suffix[:-2]
#                 with open(eval_file, "a") as f:
#                     f.write(f'{prefix}\t{suffix}\t{prediction}\t{max_score}\t1\n')

def main(context_suffix_file, completions_file, eval_file, doc_file, bert=1):
    if bert == -1:
        with open(doc_file, "r") as f:
            text = f.read()
        vectorizer = TfidfVectorizer() 
        vectorizer.fit([text])
        
        completions = load_suffix_scores(completions_file)
        with open(context_suffix_file, 'r') as f:
            for line in tqdm(f):
                line = line.replace('<eou>', '[EOU]')
                last_eou_index = line.rfind('[EOU]')  # Find the index of last <eou>
                context = line[:last_eou_index]  # Context is everything before the last <eou>
                prefix_suffix = line[last_eou_index + 5:].strip()
                prefix_suffix_parts = prefix_suffix.split('\t')
                
                if len(prefix_suffix_parts) == 2:  # Check if there are two parts (prefix and suffix)
                    prefix, suffix = prefix_suffix_parts
                    prefix = prefix.strip()
                    
                    max_score = -np.inf
                    best_suffix = ""
                    prediction = ""
                    if prefix in completions:
                        for predicted_suffix, score_str in completions[prefix]:
                            predicted_suffix, score = predicted_suffix.strip(), float(score_str.split(":")[1])
                            tfidf_matrix = vectorizer.transform([context, predicted_suffix])
                            cosine_sim = 1 - cosine(tfidf_matrix[0].toarray()[0], tfidf_matrix[1].toarray()[0])
                              
                            scaled_score = score * cosine_sim
                            if scaled_score > max_score:
                                max_score = scaled_score
                                best_suffix = predicted_suffix
                                
                        if best_suffix is not None:
                            prediction = best_suffix[len(prefix):]
                        if max_score == -np.inf:
                            max_score = 0.0
                        with open(eval_file, "a") as f:
                            f.write(f'{prefix}\t{suffix}\t{prediction}\t{max_score}\t1\n')          
                    
    else:
        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        
        # Load simCSE model
        model2 = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

        # Load completions file
        completions = load_suffix_scores(completions_file)
        # Process context and suffixes
        with open(context_suffix_file, 'r') as f:
            for line in f:
                line = line.replace('<eou>', '[EOU]')
                last_eou_index = line.rfind('[EOU]')  # Find the index of last <eou>
                context = line[:last_eou_index]  # Context is everything before the last <eou>
                prefix_suffix = line[last_eou_index + 5:].strip()
                prefix_suffix_parts = prefix_suffix.split('\t')
                
                if len(prefix_suffix_parts) == 2:  # Check if there are two parts (prefix and suffix)
                    prefix, suffix = prefix_suffix_parts
                    prefix = prefix.strip()
                    
                    max_score = -np.inf
                    best_suffix = None
                    
                    for predicted_suffix, score_str in completions[prefix]:
                        predicted_suffix, score = predicted_suffix.strip(), float(score_str.split(":")[1])
                        if bert:
                            suffix_embedding = get_bert_embeddings(predicted_suffix, model, tokenizer)
                            cosine_sim = calculate_cosine_similarity(context_embedding, suffix_embedding) - calculate_cosine_similarity(context_embedding, prefix_embedding)
                        else:
                            cosine_sim = model2.similarity(context, predicted_suffix) - model2.similarity(context, prefix)
                        
                        scaled_score = score * cosine_sim
                        if scaled_score > max_score:
                            max_score = scaled_score
                            best_suffix = predicted_suffix
                            
                    if best_suffix is not None:
                        prediction = best_suffix[len(prefix):]
                    with open(eval_file, "a") as f:
                        f.write(f'{prefix}\t{suffix}\t{prediction}\t{max_score}\t1\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute scaled scores for predicted suffixes using BERT embeddings and cosine similarity")
    parser.add_argument("--context_suffix_file", help="Path to the file containing context and suffixes")
    parser.add_argument("--completions_file", help="Path to the file containing prefix and predicted suffixes with scores")
    parser.add_argument("--eval_file", help="Path to the evaluation file")
    parser.add_argument("--doc_file", help="Path to file containing documents for tfidf")
    args = parser.parse_args()

    main(args.context_suffix_file, args.completions_file, args.eval_file, args.doc_file, bert=-1)
