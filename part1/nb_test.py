import csv
import math
from collections import defaultdict

def tokenize(text):
    """
    Same tokenizer as training:
    lowercase split on spaces.
    """
    return text.lower().split()

def load_model(model_path):
    """
    Parse model.csv
    Returns:
    - priors: dict[class] = P(class)
    - likelihoods: dict[word][class] = P(word|class)
    - classes: list of class names
    """
    priors = {}
    likelihoods = defaultdict(dict)

    with open(model_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("PP"):
            # Format: "PP\tclass,p  class,p  class,p"
            _, rest = line.split("\t", 1)
            parts = rest.split("  ")  # split on two spaces
            for p in parts:
                cls_name, prob = p.split(",")
                priors[cls_name] = float(prob)

        elif line.startswith("LP"):
            # Format: "LP\tword,class,p  word,class,p ..."
            _, rest = line.split("\t", 1)
            parts = rest.split("  ")
            for p in parts:
                word, cls_name, prob = p.split(",")
                likelihoods[word][cls_name] = float(prob)

    classes = list(priors.keys())
    return priors, likelihoods, classes

def predict_class(text, priors, likelihoods, classes):
    """
    Predict class for a single text using Naive Bayes formula:
    score(c) = log P(c) + sum_over_words[ log P(word|c) ]
    If word never appeared for that class in training, we give a tiny fallback prob.
    """
    words = tokenize(text)
    scores = {}

    for c in classes:
        # start with log prior
        score_c = math.log(priors[c])

        for w in words:
            if w in likelihoods and c in likelihoods[w]:
                score_c += math.log(likelihoods[w][c])
            else:
                # unseen word for this class
                # we just assign it a tiny probability so we don't kill the score
                score_c += math.log(1e-10)

        scores[c] = score_c

    # pick best
    best_class = max(scores, key=scores.get)
    return best_class

def run_test(model_path, test_path, output_path):
    """
    Load model + test.csv and write predictions to test_predictions.csv.
    Output columns:
    text,predicted,actual
    """
    priors, likelihoods, classes = load_model(model_path)

    rows_out = []
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            actual = row['label']
            pred = predict_class(text, priors, likelihoods, classes)
            rows_out.append({
                "text": text,
                "predicted": pred,
                "actual": actual
            })

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "predicted", "actual"])
        writer.writeheader()
        writer.writerows(rows_out)

if __name__ == "__main__":
    run_test("model.csv", "test.csv", "test_predictions.csv")
    print("test_predictions.csv generated.")
