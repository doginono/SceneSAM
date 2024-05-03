from src.conv_onet import models


def get_model(cfg,  nice=True):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.
        nice (bool, optional): whether or not use Neural Implicit Scalable Encoding. Defaults to False.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    color_grid_len = cfg['grid_len']['color']
    semantic_grid_len = cfg['grid_len']['semantic']
    hidden_size_color = cfg['model']['hidden_size']
    output_dimension_semantic = cfg['output_dimension_semantic']
    #TODO add sematic_grid_len
    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']
    if nice: #J: added argument output_dimension_semantic
        decoder = models.decoder_dict['nice'](
            dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
            middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
            color_grid_len=color_grid_len, output_dimension_semantic = output_dimension_semantic,
            semantic_grid_len= semantic_grid_len,hidden_size_color = hidden_size_color, pos_embedding_method=pos_embedding_method)
    else:
        #J:will never jumpy here in our case
        decoder = models.decoder_dict['imap'](
            dim=dim, c_dim=0, color=True,
            hidden_size=256, skips=[], n_blocks=4, pos_embedding_method=pos_embedding_method
        )
    return decoder
