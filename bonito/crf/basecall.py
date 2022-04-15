"""
Bonito CRF basecalling
"""

import torch
import torch_xla.core.xla_model as xm
import numpy as np
from crf_beam import beam_search
import pickle
# from koi.decode import beam_search, to_str
# from bonito.custom_koi_decode import beam_search

from bonito.multiprocessing import thread_iter, thread_map
from bonito.util import chunk, stitch, batchify, unbatchify, half_supported

import sys

# o = open("/home/shishirizreddy/tpu-test/outfile.log", "w")

def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    return stitch(results, size, overlap, length, stride, reverse=reverse)


# def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
#     """
#     Compute scores for model.
#     """
#     # o.write('This should be written')
#     # with torch.inference_mode():
#     print("Starting compute", file=sys.stderr)
#     device = next(model.parameters()).device
#     # device = xm.xla_device()
#     print(device, file=sys.stderr)
#     # dtype = torch.float16 if half_supported() else torch.float32
#     # scores = model(batch.to(dtype).to(device))
#     print("Sending Batch to Device", file=sys.stderr)
#     # print("Batch", file=sys.stderr)
#     # print(batch, file=sys.stderr)
#     batch = batch.to(device)
#     # print("Batch after device", file=sys.stderr)
#     # print(batch, file=sys.stderr)
#     scores = model(batch)
#     print("Sent batch to device and evaluated", file=sys.stderr)
#     if reverse:
#         print("Starting reverse complement", file=sys.stderr)
#         scores = model.seqdist.reverse_complement(scores)
#         print("Completed reverse complement", file=sys.stderr)
    
#     print("Starting beam search", file=sys.stderr)
#     # print(scores, file=sys.stderr)
#     sequence, qstring, moves = beam_search(
#         scores, beam_width=beam_width, beam_cut=beam_cut,
#         scale=scale, offset=offset, blank_score=blank_score
#     )
#     print("Completed beam search", file=sys.stderr)
#     # print(sequence, qstring, moves, file=sys.stderr)
#     return {
#         'moves': moves,
#         'qstring': qstring,
#         'sequence': sequence,
#     }

# def fmt(stride, attrs):
#     return {
#         'stride': stride,
#         'moves': attrs['moves'].numpy(),
#         'qstring': to_str(attrs['qstring']),
#         'sequence': to_str(attrs['sequence']),
#     }

# Try full CPU implementation of score computation from openvino module
def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    print("Starting compute", file=sys.stderr)
    device = next(model.parameters()).device
    # device = xm.xla_device()
    print(device, file=sys.stderr)
    # dtype = torch.float16 if half_supported() else torch.float32
    # scores = model(batch.to(dtype).to(device))
    print("Sending Batch to Device", file=sys.stderr)
    # print("Batch", file=sys.stderr)
    # print(batch, file=sys.stderr)
    batch = batch.to(device)
    # print("Batch after device", file=sys.stderr)
    # print(batch, file=sys.stderr)
    print("Sent batch to device and evaluated", file=sys.stderr)
    scores = model(batch)

    # Save scores for quicker processing
    # torch.save(scores.detach().cpu().numpy(), 'scores.pt')
    
    # Load scores
    # scores = torch.tensor(torch.load('scores.pt', map_location=device)).to(device)

    print("Starting forward", file=sys.stderr)
    fwd = model.seqdist.forward_scores(scores)
    print("Starting Backward", file=sys.stderr)
    bwd = model.seqdist.backward_scores(scores)

    posts = torch.softmax(fwd + bwd, dim=-1)

    print("Completed scoring", file=sys.stderr)
    return {
        'scores': scores.transpose(0, 1),
        'bwd': bwd.transpose(0, 1),
        'posts': posts.transpose(0, 1),
    }

def decode(x, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    print(x, file= -sys.stderr)
    print("Attempting CPU beam search", file=sys.stderr)
    sequence, qstring, moves = beam_search(x['scores'], x['bwd'], x['posts'])
    print("Completed CPU beam search", file=sys.stderr)
    return {
        'sequence': sequence,
        'qstring': qstring,
        'moves': moves,
    }

def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, reverse=False):
    """
    Basecalls a set of reads.
    """
    # print("This should be written", file=o)
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )
    # print(chunks, file=o)

    # print("Chunks", file=sys.stderr)
    # for read in reads:
    #     print((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap), file=sys.stderr)

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    # for read, batch in batches:
    #     print(read, compute_scores(model, batch, reverse=reverse), file=sys.stderr)

    # print("Scoring", file=sys.stderr)
    scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )

    print(type(scores), file=sys.stderr)

    with open('scores.pickle', 'wb') as handle:
        pickle.dump(list(scores), handle)

    with open('scores.pickle', 'rb') as handle:
        scores = (score for score in pickle.load(handle))

    # batches = thread_iter(batchify(chunks, batchsize=batchsize))
    # print("Scores", file=sys.stderr)
    # for read, batch in batches:
    #     print(read, compute_scores(model, batch, reverse=reverse), file=sys.stderr)

    # print("Completed Scoring, scores: {}".format(scores), file=sys.stderr)

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    # print("Results", file=sys.stderr)
    # for read, attrs in results:
    #     print(read, fmt(model.stride, attrs), file=sys.stderr)

    # o.write("This should be written22")
    # return thread_iter(
    #     (read, fmt(model.stride, attrs))
    #     for read, attrs in results
    # )

    print("Mapping decode to results", file=sys.stderr)

    return thread_map(decode, results, n_thread=48)
