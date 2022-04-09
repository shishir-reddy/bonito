import torch
from koi.decode import lib, ffi
from bonito.custom_koi_utils import void_ptr, empty, zeros

def beam_search(scores, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    # Try sending scores to cpu float16 from start
    scores = scores.to(torch.float16)
    scores.to(torch.device('cpu'))
    scores = scores.to(torch.float16)

    if scores.dtype != torch.float16:
        raise TypeError('Expected fp16 but received %s' % scores.dtype)

    assert(scores.is_contiguous())

    N, T, C =  scores.shape

    chunks = torch.empty((N, 4), device=scores.device, dtype=torch.int32)
    chunks[:, 0] = torch.arange(0, T * N, T)
    chunks[:, 2] = torch.arange(0, T * N, T)
    chunks[:, 1] = T
    chunks[:, 3] = 0
    chunk_results = empty((N, 8), device=scores.device, dtype=torch.int32)

    # todo: reuse scores buffer?
    aux      = empty(N * (T + 1) * (C + 4 * beam_width), device=scores.device, dtype=torch.int8)
    path     = zeros(N * (T + 1), device=scores.device, dtype=torch.int32)

    moves    = zeros(N * T, device=scores.device, dtype=torch.int8)
    sequence = zeros(N * T, device=scores.device, dtype=torch.int8)
    qstring  = zeros(N * T, device=scores.device, dtype=torch.int8)

    args = [
        void_ptr(chunks),
        chunk_results.ptr,
        N,
        void_ptr(scores),
        C,
        aux.ptr,
        path.ptr,
        moves.ptr,
        ffi.NULL,
        sequence.ptr,
        qstring.ptr,
        scale,
        offset,
        beam_width,
        beam_cut,
        blank_score,
    ]

    lib.host_back_guide_step(*args)
    lib.host_beam_search_step(*args)
    lib.host_compute_posts_step(*args)
    lib.host_run_decode(*args)

    moves = moves.data.reshape(N, -1).cpu()
    sequence = sequence.data.reshape(N, -1).cpu()
    qstring = qstring.data.reshape(N, -1).cpu()

    return sequence, qstring, moves