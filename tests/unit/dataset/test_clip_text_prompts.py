import torch

from dx_modelzoo.dataset.imagenet import _build_clip_text_inputs


class FakeModel:
    positional_embedding = torch.ones(77, 4)

    def token_embedding(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 4)


def test_builds_text_encoder_inputs_from_prompt_tokens():
    def tokenizer(texts):
        return torch.zeros(len(texts), 77, dtype=torch.long)

    result = _build_clip_text_inputs(
        FakeModel(),
        tokenizer,
        ["cat", "dog"],
        lambda name: f"a photo of a {name}",
    )

    assert result.shape == (2, 77, 4)
    assert torch.all(result == 1)
