import torch

from layers.hybrid import HVQL, HPVQL, ParallelVQL, ResQBlock, VQL


def test_hvql_uses_batch_norm_between_linear_and_activation():
    layer = HVQL(
        in_features=4,
        output_size=2,
        in_qlayer=4,
        n_qdepth=1,
        ansatz={"class": "IdentityAnsatz"},
        embedding="angle",
    )
    layer.eval()

    assert hasattr(layer, "bn")
    assert isinstance(layer.bn, torch.nn.BatchNorm1d)

    x = torch.randn(3, 4)
    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()


def test_hpvql_uses_batch_norm_between_linear_and_activation():
    layer = HPVQL(
        in_features=6,
        output_size=3,
        in_qlayer=6,
        n_qblocks=2,
        n_qdepth=1,
        ansatz={"class": "IdentityAnsatz"},
        embedding="angle",
    )
    layer.eval()

    assert hasattr(layer, "bn")
    assert isinstance(layer.bn, torch.nn.BatchNorm1d)

    x = torch.randn(4, 6)
    with torch.no_grad():
        output = layer(x)

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()


def test_vql_exposes_raw_quantum_layer():
    layer = VQL(
        in_features=4,
        n_qdepth=1,
        ansatz={"class": "IdentityAnsatz"},
        embedding="angle",
    )
    x = torch.randn(2, 4)
    with torch.no_grad():
        output = layer(x)

    assert output.shape == (2, layer.output_size)
    assert layer.output_size == layer.vqc.output_size
    assert torch.isfinite(output).all()


def test_parallel_vql_stacks_multiple_circuits():
    layer = ParallelVQL(
        in_features=5,
        n_qblocks=2,
        n_qdepth=1,
        ansatz={"class": "IdentityAnsatz"},
        embedding="angle",
    )
    x = torch.randn(3, 5)
    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, layer.output_size)
    assert layer.output_size == layer.vqc.output_size
    assert torch.isfinite(output).all()


def test_resqblock_exposes_learnable_residual_scale():
    layer = ResQBlock(
        in_features=5,
        output_size=2,
        in_qlayer=5,
        n_qdepth=1,
        ansatz={"class": "IdentityAnsatz"},
        embedding="angle",
        residual_init=0.25,
    )

    assert isinstance(layer.residual_scale, torch.nn.Parameter)
    assert torch.isclose(layer.residual_scale.detach(), torch.tensor(0.25))

    x = torch.randn(8, 5)
    layer.train()
    output = layer(x)
    output.sum().backward()

    assert layer.residual_scale.grad is not None
    assert torch.isfinite(layer.residual_scale.grad)
