"""Shape and wiring checks for the three architectures.

Smoke tests, not accuracy tests: they run on random input on CPU and catch the
class of bug that only shows up minutes into a GPU training run.
"""
import pytest

torch = pytest.importorskip("torch")

from src.models import (CNN_GRU_Binary, CNN_GRU_Classifier, InceptionBlock,
                        InceptionTime)

BATCH, STEPS, FEATURES = 4, 32, 12


class TestCnnGruClassifier:
    @pytest.mark.parametrize("bidirectional", [False, True])
    @pytest.mark.parametrize("use_attention", [False, True])
    def test_output_shape(self, bidirectional, use_attention):
        model = CNN_GRU_Classifier(FEATURES, 3, hidden_size=16, num_layers=1,
                                   bidirectional=bidirectional,
                                   use_attention=use_attention)
        out = model(torch.randn(BATCH, STEPS, FEATURES))
        assert out.shape == (BATCH, 3)

    def test_shipped_configuration_runs(self):
        """The exact configuration used for models A and B."""
        from src.config import CNN_GRU
        model = CNN_GRU_Classifier(
            FEATURES, 3, hidden_size=CNN_GRU.hidden_size,
            num_layers=CNN_GRU.num_layers, dropout_rate=CNN_GRU.dropout_rate,
            bidirectional=CNN_GRU.bidirectional,
            use_attention=CNN_GRU.use_attention)
        assert model(torch.randn(BATCH, STEPS, FEATURES)).shape == (BATCH, 3)

    def test_gradients_reach_the_first_convolution(self):
        model = CNN_GRU_Classifier(FEATURES, 3, hidden_size=16, num_layers=1)
        model(torch.randn(BATCH, STEPS, FEATURES)).sum().backward()
        grad = model.cnn[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, "no gradient signal reached the CNN front-end"

    def test_window_length_is_flexible(self):
        """A and B feed the same class windows of 96 and 200 steps."""
        model = CNN_GRU_Classifier(FEATURES, 3, hidden_size=16, num_layers=1)
        for steps in (96, 200):
            assert model(torch.randn(2, steps, FEATURES)).shape == (2, 3)

    def test_attention_actually_changes_the_output(self):
        torch.manual_seed(0)
        pooled = CNN_GRU_Classifier(FEATURES, 3, hidden_size=16, num_layers=1,
                                    use_attention=False)
        torch.manual_seed(0)
        attended = CNN_GRU_Classifier(FEATURES, 3, hidden_size=16, num_layers=1,
                                      use_attention=True)
        x = torch.randn(BATCH, STEPS, FEATURES)
        pooled.eval(); attended.eval()
        with torch.no_grad():
            assert not torch.allclose(pooled(x), attended(x))


class TestInceptionTime:
    def test_block_concatenates_four_branches(self):
        block = InceptionBlock(FEATURES, 8)
        out = block(torch.randn(BATCH, FEATURES, STEPS))
        assert out.shape == (BATCH, 8 * 4, STEPS), "expected 4 concatenated branches"

    def test_block_preserves_sequence_length(self):
        block = InceptionBlock(FEATURES, 8)
        assert block(torch.randn(2, FEATURES, 64)).shape[-1] == 64

    def test_output_shape(self):
        model = InceptionTime(FEATURES, channels=8, num_classes=3)
        assert model(torch.randn(BATCH, STEPS, FEATURES)).shape == (BATCH, 3)

    def test_handles_the_kernel_40_branch_on_short_windows(self):
        """The widest kernel is 40; padding='same' must cope with shorter input."""
        model = InceptionTime(FEATURES, channels=8, num_classes=3)
        assert model(torch.randn(2, 16, FEATURES)).shape == (2, 3)

    def test_gradients_flow(self):
        model = InceptionTime(FEATURES, channels=8, num_classes=3)
        model(torch.randn(BATCH, STEPS, FEATURES)).sum().backward()
        assert model.block1.b1.weight.grad is not None


class TestCnnGruBinary:
    def test_output_is_two_logits(self):
        model = CNN_GRU_Binary(FEATURES, input_size_static=3, hidden_size=16,
                               num_layers=1)
        out = model(torch.randn(BATCH, STEPS, FEATURES), torch.randn(BATCH, 3))
        assert out.shape == (BATCH, 2)

    def test_static_features_influence_the_output(self):
        torch.manual_seed(0)
        model = CNN_GRU_Binary(FEATURES, input_size_static=3, hidden_size=16,
                               num_layers=1).eval()
        dynamic = torch.randn(BATCH, STEPS, FEATURES)
        with torch.no_grad():
            a = model(dynamic, torch.zeros(BATCH, 3))
            b = model(dynamic, torch.ones(BATCH, 3) * 5)
        assert not torch.allclose(a, b), "the static MLP branch is not wired in"

    def test_gradients_reach_the_static_branch(self):
        model = CNN_GRU_Binary(FEATURES, input_size_static=3, hidden_size=16,
                               num_layers=1)
        model(torch.randn(BATCH, STEPS, FEATURES),
              torch.randn(BATCH, 3)).sum().backward()
        grad = model.static_mlp[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0
