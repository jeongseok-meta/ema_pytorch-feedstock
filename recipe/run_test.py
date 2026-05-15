import torch
from ema_pytorch import EMA


def test_ema_update_and_forward():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = EMA(model, beta=0.5, update_after_step=0, update_every=1)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update()
    torch.testing.assert_close(ema.ema_model.weight, torch.full_like(model.weight, 3.0))

    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update()
    assert torch.all(ema.ema_model.weight > 3.0)
    assert torch.all(ema.ema_model.weight < 5.0)

    inputs = torch.ones(1, 2)
    output = ema.forward_eval(inputs)
    torch.testing.assert_close(output, ema.ema_model(inputs))

    ema.copy_params_from_ema_to_model()
    torch.testing.assert_close(model.weight, ema.ema_model.weight)


if __name__ == "__main__":
    test_ema_update_and_forward()
