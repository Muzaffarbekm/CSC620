import csv
from collections import Counter, defaultdict

def tokenize(text):
    """
    Super simple tokenizer.
    - lowercase
    - split on spaces
    This matches the 'bag of words' assumption Naive Bayes uses.
    """
    return text.lower().split()

def train_naive_bayes(train_path):
    """
    Read training data from train.csv and compute:
    - prior P(class)
    - likelihood P(word | class) with Laplace smoothing
    Returns:
    - priors: dict[class] = P(class)
    - likelihoods: dict[(word, class)] = P(word|class)
    - vocab: set of all words
    """
    texts = []
    labels = []

    # 1. Load training data
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(row['label'])

    total_docs = len(labels)

    # 2. Prior: P(class) = (#docs in class) / (total docs)
    class_doc_counts = Counter(labels)
    priors = {c: class_doc_counts[c] / total_docs for c in class_doc_counts}

    # 3. Word counts per class
    #    word_counts_per_class[c][w] = how many times word w appeared in class c
    word_counts_per_class = defaultdict(Counter)
    total_words_per_class = Counter()
    vocab = set()

    for text, label in zip(texts, labels):
        words = tokenize(text)
        for w in words:
            vocab.add(w)
            word_counts_per_class[label][w] += 1
            total_words_per_class[label] += 1

    V = len(vocab)

    # 4. Likelihoods with Laplace smoothing:
    #    P(w|c) = (count(w,c) + 1) / (total_words_in_c + V)
    likelihoods = {}
    for c in class_doc_counts.keys():
        total_words_c = total_words_per_class[c]
        for w in vocab:
            count_wc = word_counts_per_class[c][w]
            prob_wc = (count_wc + 1) / (total_words_c + V)
            likelihoods[(w, c)] = prob_wc

    return priors, likelihoods, vocab

def write_model(model_path, priors, likelihoods):
    """
    Write the model into model.csv in the required format:
    PP  class,prior  class,prior
    LP  word,class,p(word|class)  word,class,p(word|class) ...
    """
    with open(model_path, 'w', encoding='utf-8') as f:
        # Write priors
        f.write("PP\t")
        first = True
        for cls_name, prior_val in priors.items():
            if not first:
                f.write("  ")  # two spaces as separator
            f.write(f"{cls_name},{prior_val}")
            first = False
        f.write("\n")

        # Write likelihoods
        f.write("LP\t")
        first_lp = True
        for (word, cls_name), prob_val in likelihoods.items():
            if not first_lp:
                f.write("  ")
            f.write(f"{word},{cls_name},{prob_val}")
            first_lp = False
        f.write("\n")

if __name__ == "__main__":
    priors, likelihoods, vocab = train_naive_bayes("train.csv")
    write_model("model.csv", priors, likelihoods)
    print("model.csv generated.")
