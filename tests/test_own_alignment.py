"""Our own correctness tests beyond the public smoke suite (manual Section 4.2).

Covers truncation boundaries, unusual tokenizer configurations (pad == eos, no
pad, no eos), mixed lengths, row order, prompt-only truncation, exact agreement
with a naive log-softmax loop, the DPO formula, reference no-grad, small
optimization steps, and equivalence of the memory-saving HF path.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment.batching import build_sft_batch, encode_example
from alignment.logprobs import sequence_logprobs
from alignment.losses import dpo_losses, reward_stats
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):
    """Like Qwen2.5: the pad token id equals the eos token id."""

    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    """No pad token id (the toy encode() still needs the string marker)."""

    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    """No eos token id (the toy encode() still needs the string marker)."""

    eos_token_id = None


class PositionalToyLM(torch.nn.Module):
    """Toy LM whose logits depend on the current token (so position/order matters)."""

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, input_ids, attention_mask=None, labels=None):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.table[input_ids])


def naive_logprob(model, tokenizer, prompt, response, max_length):
    ids, labels = encode_example(tokenizer, prompt, response, max_length)
    x = torch.tensor([ids])
    logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
    logp = F.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for t in range(1, len(ids)):
        if labels[t] != -100:
            total += logp[t - 1, ids[t]].item()
    return total


# ----------------------------------------------------------------------------- batching
def test_eos_kept_when_sequence_fits_exactly_and_dropped_when_one_short():
    tok = ToyTokenizer()
    prompt, response = "Q: ", "abc"
    full = len(tok.encode(prompt)) + len(tok.encode(response)) + 1
    exact = adapter.build_sft_batch(tok, [prompt], [response], max_length=full)
    assert exact["input_ids"][0, full - 1] == tok.eos_token_id
    assert exact["labels"][0, full - 1] == tok.eos_token_id
    short = adapter.build_sft_batch(tok, [prompt], [response], max_length=full - 1)
    assert short["input_ids"].shape[1] == full - 1
    assert tok.eos_token_id not in short["input_ids"][0].tolist()
    assert short["labels"][0, -1] == tok.encode("c")[0]


def test_padding_to_max_length_and_masks():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["a: ", "longer prompt: "], ["x", "yy"], max_length=24)
    assert batch["input_ids"].shape == (2, 24)
    for row in range(2):
        n = int(batch["attention_mask"][row].sum())
        assert batch["attention_mask"][row, :n].eq(1).all() and batch["attention_mask"][row, n:].eq(0).all()
        assert batch["labels"][row, n:].eq(-100).all()


def test_prompt_longer_than_max_length_gives_no_response_tokens_and_finite_losses():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    prompts, responses = ["this prompt is far too long: "], ["ok"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=8)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=8)
    assert torch.equal(logps, torch.zeros(1))
    assert torch.isfinite(adapter.compute_sft_loss(model, tok, prompts, responses, max_length=8))
    loss = adapter.compute_dpo_loss(model, ToyCausalLM(tok.vocab_size), tok, prompts, ["a"], ["b"], 0.1, 8)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_pad_equals_eos_tokenizer_keeps_eos_label_and_masks_padding():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abcd"], max_length=16)
    n_prompt = len(tok.encode("Q: "))
    assert batch["labels"][0, n_prompt + 1] == tok.eos_token_id  # eos is a target...
    assert batch["labels"][0, n_prompt + 2 :].eq(-100).all()  # ...but identical pad ids are not
    assert batch["attention_mask"][0, n_prompt + 2 :].eq(0).all()


def test_tokenizer_without_pad_token_pads_with_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["a"], max_length=10)
    n = int(batch["attention_mask"][0].sum())
    assert batch["input_ids"][0, n:].eq(tok.eos_token_id).all()


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=10)
    n = int(batch["attention_mask"][0].sum())
    assert n == len(tok.encode("Q: ")) + len(tok.encode("ab"))


def test_max_prompt_length_keeps_the_end_of_the_prompt():
    tok = ToyTokenizer()
    ids, labels = encode_example(tok, "abcdefgh", "z", max_length=20, max_prompt_length=3)
    assert ids[:3] == tok.encode("fgh")
    assert labels[:3] == [-100] * 3 and labels[3:] == tok.encode("z") + [tok.eos_token_id]


def test_dpo_batch_rows_match_individual_sft_batches():
    tok = ToyTokenizer()
    prompts, chosen, rejected = ["A: ", "Bb: "], ["good", "y"], ["bad answer", "n"]
    batch = adapter.build_dpo_batch(tok, prompts, chosen, rejected, max_length=20)
    for side, responses in (("chosen", chosen), ("rejected", rejected)):
        ref = adapter.build_sft_batch(tok, prompts, responses, max_length=20)
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(batch[side][key], ref[key])


# ----------------------------------------------------------------------------- log-probs
def test_logprobs_match_naive_loop_and_preserve_row_order():
    tok = ToyTokenizer()
    model = PositionalToyLM(tok.vocab_size)
    prompts = ["short: ", "a much longer prompt: ", "mid: "]
    responses = ["xyz", "q", "hello there"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=40)
    want = torch.tensor([naive_logprob(model, tok, p, r, 40) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    perm = [2, 0, 1]
    got_perm = adapter.compute_response_logprobs(
        model, tok, [prompts[i] for i in perm], [responses[i] for i in perm], max_length=40
    )
    torch.testing.assert_close(got_perm, want[perm], rtol=1e-5, atol=1e-5)


def test_sft_loss_is_token_mean_of_response_nll():
    tok = ToyTokenizer()
    model = PositionalToyLM(tok.vocab_size, seed=3)
    prompts, responses = ["p: ", "pp: "], ["a", "longer"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=20)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=20)
    n_tokens = len(tok.encode("a")) + 1 + len(tok.encode("longer")) + 1
    torch.testing.assert_close(loss, -logps.sum() / n_tokens)


# ----------------------------------------------------------------------------- DPO
def test_dpo_loss_matches_hand_formula():
    g = torch.Generator().manual_seed(0)
    pc, pr, rc, rr = (torch.randn(5, generator=g) * 3 for _ in range(4))
    beta = 0.3
    expected = torch.stack(
        [-torch.log(torch.sigmoid(beta * ((pc[i] - pr[i]) - (rc[i] - rr[i])))) for i in range(5)]
    ).mean()
    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)


def test_dpo_loss_is_stable_for_large_logit_gaps():
    big = torch.tensor([1e4])
    loss = adapter.compute_dpo_loss_from_logps(-big, big, big, -big, beta=1.0)
    assert torch.isfinite(loss) and loss > 1e3


def test_identical_policy_and_reference_give_zero_accuracy_and_margin():
    x = torch.tensor([-3.0, -7.0])
    _, cr, rj = dpo_losses(x, x - 1, x, x - 1, 0.1)
    stats = reward_stats(cr, rj)
    assert stats["reward_accuracy"].sum() == 0
    assert stats["reward_margin"].abs().sum() == 0


def test_reference_gets_no_gradient_and_policy_does():
    tok = ToyTokenizer()
    policy, reference = PositionalToyLM(tok.vocab_size, 1), PositionalToyLM(tok.vocab_size, 2)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["yes"], ["no"], 0.1, 16)
    loss.backward()
    assert policy.table.grad is not None and policy.table.grad.abs().sum() > 0
    assert reference.table.grad is None


def test_few_optimizer_steps_reduce_sft_and_dpo_losses():
    tok = ToyTokenizer()
    prompts, chosen, rejected = ["Q: ", "Z: "], ["yes", "ok"], ["no", "bad"]
    model = PositionalToyLM(tok.vocab_size, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first = None
    for _ in range(5):
        loss = adapter.compute_sft_loss(model, tok, prompts, chosen, 16)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert adapter.compute_sft_loss(model, tok, prompts, chosen, 16).item() < first

    policy, reference = PositionalToyLM(tok.vocab_size, 5), PositionalToyLM(tok.vocab_size, 5)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    start = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.5, 16).item()
    assert math.isclose(start, math.log(2), rel_tol=1e-6)
    for _ in range(5):
        loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.5, 16)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.5, 16).item() < start


# ----------------------------------------------------------------------------- HF fast path
@pytest.fixture(scope="module")
def tiny_qwen():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=ToyTokenizer().vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_hf_hidden_path_matches_full_logits_path_values_and_grads(tiny_qwen):
    tok = ToyTokenizer()
    batch = build_sft_batch(tok, ["Q: ", "longer question: "], ["answer", "a"], max_length=40, padding="longest")
    fast, _ = sequence_logprobs(tiny_qwen, batch, fast=True, chunk_size=3)
    fast.sum().backward()
    g_fast = tiny_qwen.model.embed_tokens.weight.grad.clone()
    tiny_qwen.zero_grad()
    slow, _ = sequence_logprobs(tiny_qwen, batch, fast=False)
    slow.sum().backward()
    g_slow = tiny_qwen.model.embed_tokens.weight.grad.clone()
    tiny_qwen.zero_grad()
    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_fast, g_slow, rtol=1e-4, atol=1e-6)


def test_hf_right_padding_does_not_change_scores(tiny_qwen):
    tok = ToyTokenizer()
    prompts, responses = ["Q: ", "a much longer question here: "], ["yes", "no"]
    with torch.no_grad():
        batched = adapter.compute_response_logprobs(tiny_qwen, tok, prompts, responses, max_length=64)
        single = torch.cat([adapter.compute_response_logprobs(tiny_qwen, tok, [p], [r], max_length=64)
                            for p, r in zip(prompts, responses)])
        naive = torch.tensor([naive_logprob(tiny_qwen, tok, p, r, 64) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched, naive, rtol=1e-5, atol=1e-5)
