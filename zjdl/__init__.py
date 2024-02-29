if __name__ == "__main__":
    import torch

    from model.model import *

    cfg = Path("cfg/cnn/_example.yaml")
    m = YamlModel(cfg)
    RepConv.reparam(m)

    print(m.main[1])

    x = m.example_input(1)
    with torch.no_grad():
        m.profile(x, repeat=3)
