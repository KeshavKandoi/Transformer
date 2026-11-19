from scripts.clean_finetune_data import HeldOutIndex, digits, negated, screen


def test_held_out_checks_both_languages_and_near_matches():
    held = HeldOutIndex([{'source': 'We are friends.', 'target': 'हम दोस्त हैं।'},
                         {'source': 'The library is open every morning from Monday through Friday.',
                          'target': 'पुस्तकालय खुला है।'}])
    assert held.overlaps('WE ARE FRIENDS!', 0) == 'exact'
    assert held.overlaps('हम दोस्त हैं?', 1) == 'exact'
    assert held.overlaps('The library is open every morning from Mondays through Friday.', 0) == 'near'
    assert held.overlaps('They bought a red bicycle.', 0) is None


def test_candidate_flags_digit_and_negation_mismatches():
    assert digits('५ apples and 10 bananas') == digits('5 सेब और १० केले')
    assert negated("We don't know.", 'en')
    assert negated('हम नहीं जानते।', 'hi')
    status, reasons = screen('We do not have 5 apples.', 'हमारे पास ७ सेब हैं।')
    assert status == 'uncertain'
    assert {'digit_mismatch_or_spelled_number', 'negation_mismatch_or_paraphrase'} <= set(reasons)


def test_passing_surface_rules_is_only_candidate_status():
    assert screen('I am going home now.', 'मैं अब घर जा रहा हूँ।') == ('high_confidence_candidate', [])
    assert screen('Home', 'घर')[0] == 'rejected'


def test_finetune_initialization_copies_exact_parent_weights():
    from dataclasses import asdict
    from types import SimpleNamespace

    import torch

    from scripts.clean_finetune import model_from
    from src.model.config import ModelConfig
    from src.model.transformer import EncoderDecoderTransformer

    config = ModelConfig(d_model=8, d_ff=16, num_heads=2, num_encoder_layers=1, num_decoder_layers=1)
    parent = EncoderDecoderTransformer(20, 0, config)
    saved = {'model_config': asdict(config), 'model_state_dict': parent.state_dict()}
    tokenizer = SimpleNamespace(vocab_size=20, pad_id=0)
    arm_a, arm_b = (model_from(saved, tokenizer) for _ in range(2))
    assert all(torch.equal(x, y) for x, y in zip(arm_a.state_dict().values(), arm_b.state_dict().values(), strict=True))
    with torch.no_grad():
        arm_a.output_projection.projection.weight.add_(1)
    assert torch.equal(arm_b.output_projection.projection.weight, parent.output_projection.projection.weight)
    assert not torch.equal(arm_a.output_projection.projection.weight, parent.output_projection.projection.weight)
