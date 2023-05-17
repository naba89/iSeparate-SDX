def get_model(model_name, model_args):
    if model_name == "WaveletHTDemucs":
        from iSeparate.models.wavelet_htdemucs.wavelet_htdemucs import WaveletHTDemucs
        model = WaveletHTDemucs(**model_args)
    elif model_name == "BSRNN":
        from iSeparate.models.bsrnn.bsrnn import BSRNN
        model = BSRNN(**model_args)
    elif model_name == "BSRNN-lite":
        from iSeparate.models.bsrnn.bsrnn_lite import BSRNN
        model = BSRNN(**model_args)
    elif model_name == "DWTTransformerUNet":
        from iSeparate.models.dwt_transformer_unet.dwt_transformer_unet import DWTTransformerUNet
        model = DWTTransformerUNet(**model_args)
    else:
        raise Exception("Unknown model!!")

    return model
