"""Free-form answers, the speech module, and the LoRA fine-tune pipeline (on a tiny local random model, no download)."""

from __future__ import annotations

import dataclasses

import pytest

from pitsense.voice import composer, guard
from pitsense.voice.facts import Facts, Neighbour, PlanF

from .test_voice import BASE, _call, _snapshot


def test_free_form_router_answers_only_from_facts():
    f = dataclasses.replace(BASE, action="STAY_OUT", fit=None)
    for q, needle in (("how far is the car ahead", "VER"), ("what is the gap behind", "NOR"), ("how old are the tyres", "18"),
                      ("what does a stop cost", "21.5"), ("where would we rejoin", "P5"), ("what is the rain chance", "40"),
                      ("how many laps are left", "36"), ("do we have a penalty", "5")):
        a = composer.compose(f, "free", q)
        assert needle in a, (q, a)
        assert guard.fact_errors(f.text("free", q), a) == []
    bare = composer.compose(Facts("STAY_OUT", "1", "AAA"), "free", "what is the rain chance")
    assert not any(ch.isdigit() for ch in bare)


def test_ask_api_without_model(monkeypatch):
    monkeypatch.setenv("PITSENSE_VOICE", "template")
    from pitsense import voice

    assert "VER" in voice.ask(_call(), _snapshot(), "How far is the car ahead?")


def test_speech_needs_a_voice_file(tmp_path, monkeypatch):
    pytest.importorskip("piper")
    from pitsense.voice import tts

    monkeypatch.setenv("PITSENSE_PIPER_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_voices"):
        tts.speak("box this lap", tmp_path / "x.wav")


def test_llm_lora_pipeline_on_a_tiny_random_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer as HFTok, models, pre_tokenizers, trainers

    from pitsense.voice import llm, synth

    try:
        synth.load_situations((2025,), None, 0)
    except FileNotFoundError:
        pytest.skip("no benchmark built")
    texts = [BASE.text("radio"), composer.radio(BASE), BASE.text("brief"), composer.brief(BASE)]
    t = HFTok(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel()
    t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<pad>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    cfg = transformers.LlamaConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
                                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    base = tmp_path / "base"
    m = transformers.LlamaForCausalLM(cfg)
    m.save_pretrained(base)
    tok.save_pretrained(base)
    tok.padding_side = "left"
    llm.train_llm(tmp_path / "voice", base=str(base), n_samples=24, per_race=20, calls=1, rank=4, tokens_per_batch=4000,
                  log=lambda *_: None, model=m, tok=tok)
    be = llm.LLMBackend(tmp_path / "voice", "cpu", base=str(base))
    a = be.decode([BASE.text("radio")], max_new=12)
    assert a == be.decode([BASE.text("radio")], max_new=12)  # greedy: deterministic
