
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    all_param_names = []
    trainable_param_names = []
    prompt_weights = 0
    prompt_normalizer = 0
    prompt_normalizer_layer = []
    soft_prompt_layers = []
    for name, param in model.named_parameters():

        all_param += param.numel()
        all_param_names.append(name)
        if param.requires_grad:
            print(name)
            if 'prompt_encoder.default.embedding' in name:
                prompt_weights+= param.numel()
                soft_prompt_layers.append(param)
            if 'prompt_normalizer' in name:
                prompt_normalizer += param.numel()
                prompt_normalizer_layer.append(param)
            trainable_params += param.numel()
            trainable_param_names.append(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {"trainable": trainable_params, "all": all_param, "trainable%": 100 * trainable_params / all_param}
