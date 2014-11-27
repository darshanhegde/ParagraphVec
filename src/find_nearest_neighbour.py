__author__ = 'darshanhegde'
"""
Reads in all the chunks of paragraph vectors generated and puts them in a single
numpy aray. Finds the nearest neighbour sentences to given sentences.
"""

import os
import sys
import numpy as np

import pickle


def pickle_load(in_file_path):
    in_file = open(in_file_path, "rU")
    data = pickle.load(in_file)
    in_file.close()
    return data


def pickle_dump(data, out_file_path):
    out_file = open(out_file_path, "w")
    pickle.dump(data, out_file)
    out_file.close()


class SentIter:

    def __init__(self, review_sent_file_path):
        self.review_sent_file_path = review_sent_file_path

    def __iter__(self):
        review_sent_file = open(self.review_sent_file_path, "rU")
        for sentence in review_sent_file:
            yield sentence.strip().split()
        review_sent_file.close()


def collect_model_files(model_folder_path, model_file_name, model_size, sentence_file_path, out_file_path):
    #count number of sentences
    SI = SentIter(sentence_file_path)
    num_sents = 0
    for sentence in SI:
        num_sents += 1
    model = np.zeros((num_sents, model_size), dtype=np.float32)
    all_files = os.listdir(model_folder_path)
    model_files = filter(lambda f_name: f_name.find(model_file_name) != -1, all_files)
    model_files_aug = [(model_file, int(model_file.split("_")[-1])) for model_file in model_files]
    model_files_aug = sorted(model_files_aug, key=lambda x: x[-1])
    model_files = [model_file for model_file, idx in model_files_aug]
    for m_idx, model_file in enumerate(model_files):
        model_chunk = pickle_load(os.path.join(model_folder_path, model_file))
        if model_chunk.shape[0] == 1024:
            model[1024*m_idx:1024*(m_idx+1), :] = model_chunk
        else:
            model[1024*m_idx:, :] = model_chunk
    pickle_dump(model, out_file_path)


class ParagraphNearestNeaighbour:

    def __init__(self, model_file_path, sentence_file_path):
        self.model = pickle_load(model_file_path)
        self.SI = SentIter(sentence_file_path)
        self.sentences = []
        for sentence in self.SI:
            self.sentences.append(sentence)

    def find_nearest_neighbours(self, test_sentece_file, out_file_path, topn):
        norm = np.linalg.norm(self.model, axis=1)
        self.model = self.model / norm[:, np.newaxis]
        test_senteces = open(test_sentece_file, "rU")
        out_file = open(out_file_path, "w")
        for test_sentence in test_senteces:
            sent_idx, sentence = test_sentence.strip().split("\t")
            print " Given sentence: ", sentence
            out_file.write(" Given: " + sentence + "\n")
            sent_idx = int(sent_idx)-1
            sent_rep = self.model[sent_idx]
            dists = np.dot(self.model, sent_rep)
            best = np.argsort(dists)[::-1][:topn+1]
            results = [(self.sentences[sim], float(dists[sim])) for sim in best]
            for sentence, score in results[1:]:
                print " ".join(sentence), score
                out_file.write(" ".join(sentence) + str(score) + "\n")
            out_file.write("\n\n")
            print "\n"
        test_senteces.close()
        out_file.close()


def main():
    collect_model_files("../model_sub/", "paragraph_model", 96, "../data/paragraph_data.txt", "../model/paragraph_model.pkl")
    PVNN = ParagraphNearestNeaighbour("../model/paragraph_model.pkl", "../data/paragraph_data.txt")
    PVNN.find_nearest_neighbours("../data/nn_sentences.txt", "../results/nn_sentences_result.txt", 5)

if __name__ == '__main__':
    main()