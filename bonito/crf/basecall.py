"""
Bonito CRF basecalling
"""

import torch
import torch_xla.core.xla_model as xm
import numpy as np
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_iter
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


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model.
    """
    # o.write('This should be written')
    # with torch.inference_mode():
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
    scores = model(batch)
    print("Sent batch to device and evaluated", file=sys.stderr)
    if reverse:
        print("Starting reverse complement", file=sys.stderr)
        scores = model.seqdist.reverse_complement(scores)
        print("Completed reverse complement", file=sys.stderr)
    
    print("Starting beam search", file=sys.stderr)
    print(scores, file=sys.stderr)
    sequence, qstring, moves = beam_search(
        scores, beam_width=beam_width, beam_cut=beam_cut,
        scale=scale, offset=offset, blank_score=blank_score
    )
    print("Completed beam search", file=sys.stderr)
    print(sequence, qstring, moves, file=sys.stderr)
    return {
        'moves': moves,
        'qstring': qstring,
        'sequence': sequence,
    }


def fmt(stride, attrs):
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': to_str(attrs['qstring']),
        'sequence': to_str(attrs['sequence']),
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
    return thread_iter(
        (read, fmt(model.stride, attrs))
        for read, attrs in results
    )
