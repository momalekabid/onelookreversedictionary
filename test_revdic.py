#!/usr/bin/env python3

"""
Runs a reverse dictionary evaluation using SentenceTransformers w/ Word2Vec backfill.
The test case file (at _TEST_CASES, below) contains a list of
(input, allowed_responses) pairs where <input> is a description
and allowed_responses is a set of permissable top-ranked
words or phrases.

Run without arguments. Results are saved to "cache" directory
to save time/cost on subsequent runs.
"""

from collections import defaultdict

import os
import json
import sys
import time
import unidecode
import numpy as np
import torch
import json
from txtai.embeddings import Embeddings
import wn
import re
import readline
import io
import spacy

_TEST_CASES = "rd-tests"
_TOPN = [1, 3, 10, 15, 100]
_MODEL_ = "reverse-dictionary-onelook"

# wn.download('oewn:2022') if not found
en = wn.Wordnet("oewn:2022")
lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # optional
embeddings = Embeddings()
embeddings.load(_MODEL_)


def load_tests():
    filedir = os.path.dirname(os.path.realpath(__file__))
    lines = open(os.path.join(filedir, _TEST_CASES)).read().split("\n")
    return lines


def normalize(word):
    """Normalize output word for comparison with eval file cases.
    Words in multi-word phrases delimited by _"""
    return unidecode.unidecode(word.lower().replace(" ", "_"))


def extract_words(text):
    word_match = re.search(r"Lemmas:\s+(.*)", text)
    if word_match:
        words_text = word_match.group(1)
        words = [normalize(word.strip()) for word in words_text.split(",")]
    else:
        words = []
    return words


def clean_results(query, words):
    """Remove duplicates, instances of the query, and lemmas found in the query"""
    parsed = lemmatizer(query)
    query_lemmas = set([normalize(token.lemma_) for token in parsed])
    query_words = set([normalize(token.text) for token in parsed])
    words = [
        w
        for w in words
        if not (
            normalize(w) in query_words
            or normalize(w) in query_lemmas
            or w == query.replace(" ", "")
            or normalize(w) == normalize(query)
        )
    ]
    words = list(dict.fromkeys(words))  # preserve order
    return words


def query_embeddings(query):
    # Search for top matches using Sentencetransformer embeddings
    results = embeddings.search(query, 100)
    words = extract_words(results[0]["text"])
    for i in range(0, 100):
        result = results[i]["text"]
        words.extend(extract_words(result))
    results = clean_results(query, words)
    return results


def main():
    # stores _TOPN value -> number of times correct result found in top n
    topn_correct = defaultdict(lambda: 0)
    latencies = []

    error_pairs = []  # cases where the top1 is wrong
    topn_error_pairs = []  # cases where not in the top(max)

    filedir = os.path.dirname(os.path.realpath(__file__))
    lines = open(os.path.join(filedir, _TEST_CASES)).read().split("\n")
    for line in lines:
        timer_start = time.time()
        row = line.strip().split("\t")
        query, expected = row[0], row[-1]
        query = query.strip()
        expected = expected.strip()
        if len(query) == 0:
            continue
        try:
            result = query_embeddings(query)
        except KeyError:
            pass
        latency = (time.time() - timer_start) * 1000.0
        latencies.append(latency)

        my_results = [normalize(result[i]) for i in range(len(result))]
        expected_results = [normalize(w) for w in expected.split("|")]
        top_result = "<none>" if len(result) < 1 else my_results[0]

        outcome = "WRONG"
        for i in range(min(max(_TOPN), len(my_results))):
            if my_results[i] in expected_results:
                for topn in _TOPN:
                    if i < topn:
                        topn_correct[topn] += 1
                if i == 0:
                    outcome = "CORRECT"
                else:
                    outcome = "IN_TOP_%d" % (topn)
                break
            elif i == 0:
                error_pairs.append((query, expected, top_result))

        if outcome == "WRONG":
            topn_error_pairs.append((query, expected, top_result))

        print(
            '%s: Queried for "%s", got %s, expected %s'
            % (
                outcome,
                query,
                [normalize(a) for a in result][: _TOPN[-1]],
                expected,
            )
        )

    latencies.sort()
    for topn in _TOPN:
        rate = topn_correct[topn] / len(lines)
        print(
            "Accuracy @top%d:  %d / %d = %f"
            % (
                topn,
                topn_correct[topn],
                len(lines),
                rate,
            )
        )
    with open("top1_errors.out", "w") as fs:
        print(
            "".join(
                [
                    '\n"%s" (expected %s, got %s)' % (query, correct, wrong)
                    for (query, correct, wrong) in error_pairs
                ]
            ),
            file=fs,
        )
    with open("top%d_errors" % (max(_TOPN)), "w") as fs:
        print(
            "".join(
                [
                    '\n"%s" (expected %s, got %s)' % (query, correct, wrong)
                    for (query, correct, wrong) in topn_error_pairs
                ]
            ),
            file=fs,
        )

    print(
        "Mean/median/95%%ile latency: %f,%f,%f msec"
        % (
            sum(latencies) / len(lines),
            latencies[len(latencies) // 2],
            latencies[int(len(latencies) * 0.95)],
        )
    )


def normalize(word):
    """Normalize output word for comparison with eval file cases.
    Words in multi-word phrases delimited by _"""
    return unidecode.unidecode(word.lower().replace(" ", "_"))


if __name__ == "__main__":
    main()
