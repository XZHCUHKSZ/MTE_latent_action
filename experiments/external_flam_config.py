"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import hashlib

from pathlib import Path

def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def author_config(c):
    from configs.config import Config
    from configs.misc.trainer_cfg import TrainerConfig, DataloaderConfig
    cfg = Config(update_tokenizer=True, update_lam=False, distributed=False, seed=c['seed'],
                 trainer=TrainerConfig(dataloader=DataloaderConfig()))
    t = cfg.tokenizer
    t.image_size = 128
    t.encoder_type, t.decoder_type, t.quantizer_type = 'impala', 'lapo', 'fsq'
    t.codebook_size, t.z_channels, t.codebook_dim = 1024, 128, 128
    t.quantizer_all.fsq.codebook_levels = [4]*5
    t.quantizer_all.fsq.__post_init__()
    t.__post_init__()
    t.overwrite_children_cfg()
    t.batch_size, t.learning_rate = c['tokenizer_batch'], c['tokenizer_learning_rate']
    t.image_loss.eval_fvd = False
    assert t.image_loss.lpips_weight == c['tokenizer_lpips_weight']
    cfg.update_tokenizer, cfg.update_lam, cfg.lam_type = False, True, 'factored'
    l = cfg.lam_all.factored
    l.image_size, l.d_model, l.num_blocks, l.num_heads = 128, 256, 2, 8
    l.sub_traj_len_train = l.sub_traj_len_valid = c['sequence_length']
    l.encoder_type, l.decoder_type, l.tokenizer_quantizer_type = 'impala', 'lapo', 'fsq'
    l.z_channels, l.tokenizer_codebook_size, l.tokenizer_codebook_dim = 128, 1024, 128
    l.tokenizer_quantizer_all.fsq.codebook_levels = [4]*5
    l.tokenizer_quantizer_all.fsq.__post_init__()
    l.num_slots, l.history_len, l.num_prediction_steps = 4, 1, 5
    l.factorizer_type, l.aggregator_type = 'temporal_slot_attn', 'block_transformer'
    factor = l.factorizer_all.temporal_slot_attn
    factor.slot_init_type, factor.slot_update_type, factor.use_temporal_attn = 'learn_each', 'slot_attn', True
    l.aggregator_all.block_transformer.use_cross_attn = False
    l.aggregator_all.block_transformer.add_slot_embedding = False
    l.factored_idm.idm_type, l.factored_idm.use_film = 'spatial_shifted_temporal', False
    l.fdm.fdm_type, l.fdm.predict_next_slot = 'cross_spatiotemporal', True
    l.use_slot_pred_adapter, l.use_flow_matching, l.use_z_pred_adapter = True, False, True
    l.action_quantizer_type, l.action_num_codebooks, l.action_codebook_size = 'vae_vq', 1, 4
    a = l.action_quantizer_all.vae_vq.kl_coefficient_annealing
    a.annealing_start = a.annealing_end = a.value_start = 0.0
    a.value_end = c['vae_kl_coefficient']
    l.batch_size, l.learning_rate, l.weight_decay, l.grad_norm_clip = c['factored_batch'], .0001, 0.0, 1.0
    l.slot_loss_type, l.slot_loss_weight, l.image_loss_weight = 'mse', 0.0, 0.0
    l.image_loss.eval_fvd = False
    l.__post_init__()
    cfg.__post_init__()
    assert cfg.lam.action_quantizer.codebook_dim == 64
    assert cfg.tokenizer.quantizer.codebook_size == cfg.lam.tokenizer_quantizer.codebook_size == 1024
    return cfg
