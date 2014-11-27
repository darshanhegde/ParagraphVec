ParagraphVecCUDA
================

Project submission for CSCI-GA.3033-004 Graphics Processing Units (GPUs): Architecture and Programming.

CUDA implementation of paragraph vector algorithm.

Just implements PV-DBOW Model (Paragraph Vector Distributed Bag of Words). 

Expects that you have already trained the word-vectors using gensim version of word2vec.

However, I plan to implement a version which would just take any corpus and generate both 
word vectors and document vectors.


Instructions for running on NYU CUDA Cluster
============================================

Copy the implementation to appropriate directory.

$git clone https://github.com/darshanhegde/ParagraphVec

I've run all my experiments on cuda1 cluster. I'm using PyCUDA for my implementation. 
PyCUDA on cuda1 uses CUDA SDK version of 5.5. 

To load CUDA SDK 5.5

$module load cuda-5.5

I'm using floating point version of AtomicAdd which requires CUDA device with compute capability 2.x and higher. 
I tested on device1 because device0 was mostly busy ! So to make sure PyCUDA uses device 1 export following environment variable.

$export CUDA_DEVICE=1

Now, we are ready to lauch the paragraph vector code. Change your directory to ./ParagraphVec/src of the project.

$python paragraph_vector.py gpu 512

Once the paragraph vectors are trained. We can check the correctness by launching the nearest neighbor script.

$python find_nearest_neighbour.py

This script will printout 20 test sentences and their nearest neighbors. As long as nearest neighbors semantically make sense we have everything running fine. Notice that, paragraph vector algorithm finds nearest neighbors in a semantic sense and has non-trivial generalization as explained in the writeup. 




