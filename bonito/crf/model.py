"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np

from koi.ctc import SequenceDist, Max, Log, semiring
from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict

from collections import namedtuple

def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1

def grad(f, x):
    x = x.detach().requires_grad_()
    with torch.enable_grad():
        y = f(x)
    return torch.autograd.grad(y, x)[0].detach()

def max_grad(x, dim=0):
    return torch.zeros_like(x).scatter_(dim, x.argmax(dim, True), 1.0)

semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum', 'dsum'))
Log = semiring(zero=-1e38, one=0., mul=torch.add, sum=torch.logsumexp, dsum=torch.softmax)
Max = semiring(zero=-1e38, one=0., mul=torch.add, sum=(lambda x, dim=0: torch.max(x, dim=dim)[0]), dsum=max_grad)

def scan(Ms, idx, v0, S:semiring=Log):
    T, N, C, NZ = Ms.shape
    alpha = Ms.new_full((T + 1, N, C), S.zero)
    alpha[0] = v0
    for t in range(T):
        alpha[t+1] = S.sum(S.mul(Ms[t], alpha[t, :, idx]), dim=-1)
    return alpha
'''
Add in OPENVINO LogZ CPU implementation to bypass CUDA reqs
'''
class LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ms, idx, v0, vT, scan, S:semiring):
        alpha = scan(Ms, idx, v0, S)
        ctx.save_for_backward(alpha, Ms, idx, vT)
        ctx.semiring, ctx.scan = S, scan
        return S.sum(S.mul(alpha[-1], vT), dim=1)

    @staticmethod
    def backward(ctx, grad):
        alpha, Ms, idx, vT = ctx.saved_tensors
        S, scan = ctx.semiring, ctx.scan
        T, N, C, NZ = Ms.shape
        idx_T = idx.flatten().argsort().reshape(*idx.shape)
        Ms_T = Ms.reshape(T, N, -1)[:, :, idx_T]
        idx_T = torch.div(idx_T, NZ, rounding_mode='floor')
        beta = scan(Ms_T.flip(0), idx_T, vT, S)
        g = S.mul(S.mul(Ms.reshape(T, N, -1), alpha[:-1, :, idx.flatten()]).reshape(T, N, C, NZ), beta[:-1, :, :, None].flip(0))
        g = S.dsum(g.reshape(T, N, -1), dim=2).reshape(T, N, C, NZ)
        return grad[None, :, None, None] * g, None, None, None, None, None

class CTC_CRF(SequenceDist):

    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        print("Beginning LogZ")
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        v0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        vT = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return LogZ.apply(Ms, self.idx.to(torch.int64), v0, vT, scan, S)
        # return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        print("Normalizing scores")
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        v0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return scan(Ms, self.idx.to(torch.int64), v0, S)

        # return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        vT = scores.new_full((N, self.n_base**(self.state_len)), S.one)
        idx_T = self.idx.flatten().argsort().reshape(*self.idx.shape)
        Ms_T = scores[:, :, idx_T]
        idx_T = torch.div(idx_T, self.n_base + 1, rounding_mode='floor')
        return scan(Ms_T.flip(0), idx_T.to(torch.int64), vT, S).flip(0)

        # return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        print("Starting reverse complement")
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        print("Finished reverse complement")
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        print("Starting CTC score preparation")
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)
        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        print("Starting CTC loss")
        if normalise_scores:
            scores = self.normalise(scores)
        print("Score normalization success")
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        print("Successfully Prepared CTC scores")
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        print("LogZ Success")
        loss = - (logz / target_lengths)
        if loss_clip:
            print("Starting loss clip")
            loss = torch.clamp(loss, 0.0, loss_clip)
            print("Loss clip success")
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    print("Attempting Conv: {};{}".format(c_in,c_out))
    o = Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)
    print("Conv Success")
    return o
    # return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)


def rnn_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5):
    rnn = layers[rnn_type]
    return Serial([
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
            Permute([2, 0, 1]),
            *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
            LinearCRFEncoder(
                features, n_base, state_len, activation='tanh', scale=scale,
                blank_score=blank_score, expand_blanks=expand_blanks
            )
    ])


class SeqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        print("Attempting forward")
        f = self.encoder(x)
        print("Completed forward")
        return f
        # return self.encoder(x)

    def decode_batch(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
        else: #old-style
            encoder = rnn_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])
        super().__init__(encoder, seqdist)
        self.config = config
