import torch


def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, path="model.pth"):
    model = model_class()
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
