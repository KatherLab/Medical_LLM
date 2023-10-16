#%%
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from gensim.models import Word2Vec
from pyemd import emd
#%%
reference_text = "ASCO's clinical practice guidelines provide evidence-based recommendations and outline appropriate methods of treatment and care for clinicians. Our guidelines address specific clinical situations (disease-oriented) or the use of approved medical products, procedures, or tests (modality-oriented)."
generated_text = "The ASCO guideline is a document published by ASCO to assist providers in clinical decision making. It provides guidance and recommendations for the treatment and care of patients. However, it should not be relied upon as being complete or accurate, nor should it be considered as inclusive of all proper treatments or methods of care or as a statement of the standard of care."

#%%
def bleu_eval(reference, candidate):
    references = [reference]
    candidates = [candidate]
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, candidates, smoothing_function=smoothie)
    return bleu_score

#%%
def rouge_eval(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


#%%
def cider_eval(reference, candidate):
    references = [reference.split()]
    candidates = [candidate.split()]
    model = Word2Vec(references + candidates, min_count=1, size=100)
    cider_score = model.wmdistance(reference.split(), candidate.split())
    return cider_score
#%%
references_tokens = [nltk.word_tokenize(reference_text)]
model = Word2Vec(references_tokens, min_count=1, size=100)
def wmd_eval(reference, candidate):
    word_vectors = model.wv
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.words_tokenize(candidate)
    distance_matrix = word_vectors.distances(reference_tokens, candidate_tokens)
    wmd_score = emd(reference_tokens, candidate_tokens, distance_matrix)
    return wmd_score

#%%
def meteor_eval(reference, candidate):
    meteor_score_value = meteor_score([reference], candidate)
    return meteor_score_value


#%%
bleu_score = bleu_eval(reference_text, generated_text)
rouge_scores = rouge_eval(reference_text, generated_text)
#cider_score = cider_eval(reference_text, generated_text)
#wmd_score = wmd_eval(reference_text, generated_text)


print(f"BLEU Score: {bleu_score}")
print("ROUGE Scores:")
for metric, score in rouge_scores.items():
    print(f"{metric}: {score}")
#print(f"METEOR Score: {meteor_score}")
#print(f"CIDEr Score: {cider_score}")
#print(f"Word Mover's Distance Score: {wmd_score}")

# %%
