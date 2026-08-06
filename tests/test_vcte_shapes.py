import torch

from vlff_net import VCTENeckModulator, VCTETextAdapter, VCTETextEnhancer, VCTEVisualAdapter


def test_released_vcte_pipeline_shapes() -> None:
    batch, classes, text_dim = 2, 6, 512
    text = torch.randn(classes, text_dim)
    visual = torch.randn(batch, 256, 32, 32)

    adapted = VCTETextAdapter(text_dim, bottleneck=64)(text)
    visual_context = VCTEVisualAdapter(256, out_dim=text_dim)(visual)
    enhanced, text_context = VCTETextEnhancer(text_dim, text_dim, text_dim, rank=64)(
        adapted, visual_context
    )
    modulated = VCTENeckModulator(128, text_dim=text_dim)(
        torch.randn(batch, 128, 64, 64), text_context
    )

    assert adapted.shape == (classes, text_dim)
    assert visual_context.shape == (batch, text_dim)
    assert enhanced.shape == (batch, classes, text_dim)
    assert text_context.shape == (batch, text_dim)
    assert modulated.shape == (batch, 128, 64, 64)
    assert torch.isfinite(modulated).all()

