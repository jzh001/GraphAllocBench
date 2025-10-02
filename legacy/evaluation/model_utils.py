from constants import RL_Model, INFERENCE_DEVICE

def load_model(model_path, env=None):
    model = RL_Model.load(model_path, env=env, device=INFERENCE_DEVICE)
    return model