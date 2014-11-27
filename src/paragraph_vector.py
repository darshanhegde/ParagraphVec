"""
Implements PV-DBOW Model on CUDA.
"""

__author__ = 'darshanhegde'

import numpy as np
import pickle
import random
import sys

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#initialize the device
from pycuda import autoinit


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


class WordRep(object):
    """
    This is used to store representation of word as a path from root to leaf.
    length: length of the path from root to leaf.
    code: sequence of {+1, -1} +1 if left child, -1 if right child.
    point: sequence of indices into syn1 matrix.
    """
    memsize = 8 + np.intp(0).nbytes + np.intp(0).nbytes

    def __init__(self, code, point, struct_ptr):
        self.code = cuda.to_device(code)
        self.point = cuda.to_device(point)
        self.code_shape, self.code_dtype = code.shape, code.dtype
        self.point_shape, self.point_dtype = point.shape, point.dtype
        cuda.memcpy_htod(int(struct_ptr), np.int32(code.size))
        cuda.memcpy_htod(int(struct_ptr) + 8, np.intp(int(self.code)))
        cuda.memcpy_htod(int(struct_ptr) + 8 + np.intp(0).nbytes, np.intp(int(self.point)))

    def __str__(self):
        return "len: " + str(self.code_shape) + " code: " + str(cuda.from_device(self.code, self.code_shape, self.code_dtype)) + \
            " point: " + str(cuda.from_device(self.point, self.point_shape, self.point_dtype))


class Dictionary:

    def __init__(self, iterator):
        self.iterator = iterator
        self.id2token = {}
        self.token2id = {}
        self.create_index()

    def create_index(self):
        for idx, token in enumerate(self.iterator):
            self.token2id[token] = idx
            self.id2token[idx] = token

    def __contains__(self, token):
        return token in self.token2id

    def __len__(self):
        return len(self.token2id)

    def items(self):
        """
        this may be is worst way to implement items method.
        """
        return [(idx, token) for idx, token in self.id2token.items()]


class ParagraphVector:
    """
    A dummy paragraph vector class, which is test-case for original code.
    """

    def __init__(self, sentence_file_path, syn1_file_path, words_rep_file_path, model_file_path, num_iter=50,
                 alpha=0.05, size=96, gpu=True, batch_size=1024, max_batches=4):
        self.SI = SentIter(sentence_file_path)
        self.syn1 = pickle_load(syn1_file_path)
        self.words_rep = pickle_load(words_rep_file_path)
        self.model_file_path = model_file_path
        print "Done loading Word2Vec files."
        self.dictionary = Dictionary(self.words_rep.keys())
        self.size = size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.kernel_str = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        #define SIZE 96
        #define MAX_SHARED_LEN 30

        typedef struct{
                int len, __padding;
                int* code;
                int* point;
        }WordRep;

        __device__ float sigmoid(float val){
            return 1.0F / (1.0F + expf(-1.0F * val));
        }

        __global__ void train_sg(float* sentence_reps, float alpha, int* words, WordRep* base_word_rep,
                        float* syn1){
            // set-up indexes
            int sent_idx = blockIdx.x;
            int dim_idx = threadIdx.x;
            int word_idx = words[sent_idx];
            WordRep* word_rep = base_word_rep+word_idx;
            int code_len = word_rep->len;
            int* code = word_rep->code;
            int* point = word_rep->point;

            // load the sentence vector to shared memory.
            __shared__ float sentence_rep[SIZE];
            __shared__ float gradient[SIZE];
            __shared__ float dot_product[MAX_SHARED_LEN];
            int sentence_rep_idx = sent_idx*SIZE + dim_idx;
            sentence_rep[dim_idx] = sentence_reps[sentence_rep_idx];
            gradient[dim_idx] = 0;
            if(dim_idx < MAX_SHARED_LEN){
                dot_product[dim_idx] = 0;
            }

            __syncthreads();

            // Compute the dot product needed for soft-max function
            float* syn1_offset;
            for(int c_idx=0; c_idx<code_len; c_idx++){
                syn1_offset = syn1 + point[c_idx]*SIZE;
                atomicAdd(&dot_product[c_idx], syn1_offset[dim_idx] * (sentence_rep[dim_idx])) ;
            }

            __syncthreads();

            // Compute the gradient vector
            for(int c_idx=0; c_idx<code_len; c_idx++){
                syn1_offset = syn1 + point[c_idx]*SIZE;
                atomicAdd(&gradient[dim_idx], (1-sigmoid(code[c_idx]*dot_product[c_idx])) * code[c_idx] * syn1_offset[dim_idx]);
            }

            __syncthreads();

            // Subtract the gradient vector
            sentence_rep[dim_idx] -= (alpha*gradient[dim_idx]);

            // Write back to global memory.
            sentence_reps[sentence_rep_idx] = sentence_rep[dim_idx];
        }
        """)
        if gpu:
            print "Running on GPU"
            self.train = self.train_gpu
        else:
            print "Running on CPU"
            self.train = self.train_cpu
        # Process in batch of 1024 sentences.
        self.sentences = []
        self.batch = 0
        for sentence in self.SI:
            self.sentences.append(sentence)
            if len(self.sentences) == self.batch_size:
                print "Processing batch: ", self.batch
                self.init_sent_vec()
                self.train(self.num_iter, self.model_file_path + "_" + str(self.batch))
                self.batch += 1
                if self.batch > max_batches:
                    print " Done with batch: ", self.batch-1, " returning."
                    return
                self.sentences = []
        self.init_sent_vec()
        self.train(self.num_iter, self.model_file_path + "_" + str(self.batch))
        self.batch += 1

    def init_sent_vec(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        self.num_sents = len(self.sentences)
        self.sent_reps = np.empty((self.num_sents, self.size), dtype=np.float32)
        for idx in xrange(self.num_sents):
            self.sent_reps[idx] = (np.random.rand(self.size) - 0.5) / self.size

    def train_gpu(self, num_iter, model_file_path):
        if self.batch == 0:
            # Prepare to send the numpy array to gpu
            self.syn1_gpu = cuda.to_device(self.syn1)

            # Create word idx and related data-structure.
            self.base_word_rep = cuda.mem_alloc(len(self.dictionary)*WordRep.memsize)
            word_rep_ptr = int(self.base_word_rep)
            self.word_reps = {}
            for w_idx, word in sorted(self.dictionary.items()):
                word_code = 1-2*self.words_rep[word][0].astype(dtype=np.int32)
                word_point = self.words_rep[word][1].astype(dtype=np.int32)
                self.word_reps[w_idx] = WordRep(word_code, word_point, word_rep_ptr)
                word_rep_ptr += WordRep.memsize
            print "GPU transfers done."


        self.sent_reps_gpu = cuda.to_device(self.sent_reps)
        # Prepare sentences for GPU transfer.
        idx_sentences = [[self.dictionary.token2id[word] for word in sentence if word in self.dictionary]
                         for sentence in self.sentences]

        # Prepare the kernel function
        kernel = self.kernel_str.get_function("train_sg")
        words = np.empty(self.num_sents, dtype=np.int32)
        # sent_reps = np.copy(self.sent_reps)
        for iter in range(num_iter):
            # Sample words for each sentence and transfer to GPU
            for s_idx in range(self.num_sents):
                words[s_idx] = random.choice(idx_sentences[s_idx])
            words_gpu = cuda.to_device(words)
            kernel(self.sent_reps_gpu, np.float32(self.alpha), words_gpu, self.base_word_rep, self.syn1_gpu,
                   block=(self.size, 1, 1), grid=(self.num_sents, 1, 1))
            # autoinit.context.synchronize()
        self.sent_reps = cuda.from_device(self.sent_reps_gpu, self.sent_reps.shape, self.sent_reps.dtype)
        pickle_dump(self.sent_reps, model_file_path)

    def train_cpu(self, num_iter, model_file_path):
        # Using this to avoid picking OOV words.
        idx_sentences = [[self.dictionary.token2id[word] for word in sentence if word in self.dictionary]
                         for sentence in self.sentences]
        for s_idx in range(self.num_sents):
            for iter in range(num_iter):
                word_idx = random.choice(idx_sentences[s_idx])
                word = self.dictionary.id2token[word_idx]
                gradient = self.compute_gradient(self.sent_reps[s_idx], word)
                self.sent_reps[s_idx] -= (self.alpha*gradient)
        pickle_dump(self.sent_reps, model_file_path)

    def compute_gradient(self, sentence_rep, word):
        """
        Computes the gradient of cost-function w.r.t sentence_rep for a
        given word cost-function
        :param sentence_rep: present representation of sentence.
        :param word: word w.r.t which gradient has to be computed.
        :return: gradient vector.
        """
        gradient = np.zeros(self.size, dtype=np.float32)
        l2a = self.syn1[self.words_rep[word][1]]                  # 2d matrix, codelen x layer1_size
        fa = 1.0 / (1.0 + np.exp(-np.dot(sentence_rep, l2a.T)))   # propagate hidden -> output
        ga = (1 - self.words_rep[word][0] - fa)                   # vector of error gradients
        gradient += np.dot(ga, l2a)                               # save error
        return gradient

    def find_closest(self, sentence_idx, topn):
        """
        Finds the nearest neighbor sentences to given sentence index.
        Since the processing is done in batches, this finds nearest neighbors in the last batch only.
        Don't use this to find nearest neighbors in general. Just a test function to make sure everthing is
        running fine.
        """
        # Normalize word vectors.
        print "Asked sentence: ", " ".join(self.sentences[sentence_idx])
        sent_reps = np.copy(self.sent_reps)
        norm = np.linalg.norm(sent_reps, axis=1)
        sent_reps = sent_reps / norm[:, np.newaxis]
        sent_rep = sent_reps[sentence_idx]
        dists = np.dot(sent_reps, sent_rep)
        best = np.argsort(dists)[::-1][:topn]
        results = [(self.sentences[sim], float(dists[sim])) for sim in best]
        return results


def main():
    if len(sys.argv) != 3:
        print "Usage: python dummy_project.py <cpu/gpu> <max_batches>",
        return
    gpu = True
    if sys.argv[1] == "cpu":
        gpu = False
    max_batches = int(sys.argv[2])
    print "Running on", gpu, " with max-batch size: ", max_batches
    PV = ParagraphVector("../data/paragraph_data.txt", "../model/syn1.pkl", "../model/words_rep.pkl", "../model_sub/paragraph_model",
                         num_iter=200, gpu=gpu, max_batches=max_batches)
    results = PV.find_closest(sentence_idx=24, topn=5)
    for result, score in results:
        print " ".join(result), score

if __name__ == '__main__':
    main()