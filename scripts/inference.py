"""
Run single-sample CFM inference with a VAE decoder and save the forecast as GIFs.
"""

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from omegaconf import OmegaConf
from torchdiffeq import odeint_adjoint as odeint

from diffusers.models.autoencoders import AutoencoderKL

from pkcast.models.cuboid_transformer_unet import CuboidTransformerUNet
from pkcast.data.sevir import (
    DynamicSequentialSevirDataset,
    post_process_samples,
)
from pkcast.visualization.cartopy import make_animation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the trained VAE and CFM models, run inference, and save GIFs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pkcast.yaml",
    )
    parser.add_argument(
        "--cfm_checkpoint",
        type=str,
        default=(
            "artifacts/sevir/pkcast/"
            "2026-04-22_11-20-07_cfm_pkcast_ddp_14b2d14f_main/"
            "models/early_stopping_model.pt"
        ),
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Overrides config.autoencoder_params.autoencoder_checkpoint if set.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="datasets/sevir/data/sevir_full/nowcast_testing_full.h5",
    )
    parser.add_argument(
        "--test_meta",
        type=str,
        default="datasets/sevir/data/sevir_full/nowcast_testing_full_META.csv",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--probabilistic_samples", type=int, default=1)
    parser.add_argument("--euler_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/sevir/inference_gif",
    )
    parser.add_argument(
        "--cartopy_features",
        action="store_true",
        help="Enable states/rivers/lakes overlays in GIFs.",
    )
    return parser.parse_args()


def safe_encode(model, x):
    return model.encode(x)


def safe_decode(model, x):
    return model.decode(x)


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def save_gif(frames, metadata, output_path, title, fps, cartopy_features):
    fig = plt.figure()
    anim = make_animation(
        frames,
        metadata,
        title=title,
        fig=fig,
        cartopy_features=cartopy_features,
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def build_vae(config, checkpoint_path, device):
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=list(config.autoencoder_params.down_block_types),
        up_block_types=list(config.autoencoder_params.up_block_types),
        block_out_channels=list(config.autoencoder_params.block_out_channels),
        act_fn=config.autoencoder_params.act_fn,
        latent_channels=config.autoencoder_params.latent_channels,
        norm_num_groups=config.autoencoder_params.norm_num_groups,
        layers_per_block=config.autoencoder_params.layers_per_block,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    model = model.to(device)
    model.eval()
    return model


def build_cfm(config, checkpoint_path, device, input_shape_pkcast, output_shape_pkcast):
    model_config = OmegaConf.to_object(config.latent_model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CuboidTransformerUNet(
        input_shape=input_shape_pkcast,
        target_shape=output_shape_pkcast,
        base_units=model_config["base_units"],
        block_units=None,
        scale_alpha=model_config["scale_alpha"],
        num_heads=model_config["num_heads"],
        attn_drop=model_config["attn_drop"],
        proj_drop=model_config["proj_drop"],
        ffn_drop=model_config["ffn_drop"],
        downsample=model_config["downsample"],
        downsample_type=model_config["downsample_type"],
        upsample_type=model_config["upsample_type"],
        upsample_kernel_size=model_config["upsample_kernel_size"],
        depth=model_config["depth"],
        block_attn_patterns=[model_config["self_pattern"]] * len(model_config["depth"]),
        num_global_vectors=model_config["num_global_vectors"],
        use_global_vector_ffn=model_config["use_global_vector_ffn"],
        use_global_self_attn=model_config["use_global_self_attn"],
        separate_global_qkv=model_config["separate_global_qkv"],
        global_dim_ratio=model_config["global_dim_ratio"],
        ffn_activation=model_config["ffn_activation"],
        gated_ffn=model_config["gated_ffn"],
        norm_layer=model_config["norm_layer"],
        padding_type=model_config["padding_type"],
        checkpoint_level=model_config["checkpoint_level"],
        pos_embed_type=model_config["pos_embed_type"],
        use_relative_pos=model_config["use_relative_pos"],
        self_attn_use_final_proj=model_config["self_attn_use_final_proj"],
        attn_linear_init_mode=model_config["attn_linear_init_mode"],
        ffn_linear_init_mode=model_config["ffn_linear_init_mode"],
        ffn2_linear_init_mode=model_config["ffn2_linear_init_mode"],
        attn_proj_linear_init_mode=model_config["attn_proj_linear_init_mode"],
        conv_init_mode=model_config["conv_init_mode"],
        down_linear_init_mode=model_config["down_up_linear_init_mode"],
        up_linear_init_mode=model_config["down_up_linear_init_mode"],
        global_proj_linear_init_mode=model_config["global_proj_linear_init_mode"],
        norm_init_mode=model_config["norm_init_mode"],
        time_embed_channels_mult=model_config["time_embed_channels_mult"],
        time_embed_use_scale_shift_norm=model_config["time_embed_use_scale_shift_norm"],
        time_embed_dropout=model_config["time_embed_dropout"],
        unet_res_connect=model_config["unet_res_connect"],
        mean=checkpoint["mean"],
        std=checkpoint["std"],
    )
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    vae_checkpoint = (
        args.vae_checkpoint or config.autoencoder_params.autoencoder_checkpoint
    )

    if not os.path.exists(args.cfm_checkpoint):
        raise FileNotFoundError(f"CFM checkpoint not found: {args.cfm_checkpoint}")
    if not os.path.exists(vae_checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lag_time = int(config.data_params.lag_time)
    lead_time = int(config.data_params.lead_time)
    time_spacing = int(config.data_params.time_spacing)
    normalized_autoencoder = bool(config.autoencoder_params.normalized_autoencoder)
    batch_size_autoencoder = config.test_params.batch_size_autoencoder
    euler_steps = (
        int(args.euler_steps)
        if args.euler_steps is not None
        else int(config.test_params.euler_steps)
    )

    dataset = DynamicSequentialSevirDataset(
        meta_csv=args.test_meta,
        data_file=args.test_file,
        data_type="vil",
        raw_seq_len=49,
        lag_time=lag_time,
        lead_time=lead_time,
        time_spacing=time_spacing,
        stride=12,
        channel_last=False,
        debug_mode=False,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} is out of range for dataset of size {len(dataset)}"
        )

    x_cond, x_true, metadata = dataset[args.sample_index]
    x_cond = x_cond.unsqueeze(0)
    x_true = x_true.unsqueeze(0)

    vae_model = build_vae(config, vae_checkpoint, device)

    with torch.no_grad():
        x_cond_device = x_cond.to(device)
        bsz, channels, t_in, height, width = x_cond_device.shape
        x_cond_2d = x_cond_device.permute(0, 2, 1, 3, 4).reshape(
            bsz * t_in, channels, height, width
        )
        if normalized_autoencoder:
            x_cond_2d = x_cond_2d / 255.0
        encoded_obj = safe_encode(vae_model, x_cond_2d)
        x_cond_latent = encoded_obj.latent_dist.mode()

    latent_channels = x_cond_latent.shape[1]
    latent_h = x_cond_latent.shape[2]
    latent_w = x_cond_latent.shape[3]
    x_cond_latent = x_cond_latent.reshape(
        bsz, t_in, latent_channels, latent_h, latent_w
    ).permute(0, 2, 1, 3, 4)

    input_shape_pkcast = (t_in, latent_h, latent_w, latent_channels)
    output_shape_pkcast = (x_true.shape[2], latent_h, latent_w, latent_channels)

    cfm_model = build_cfm(
        config,
        args.cfm_checkpoint,
        device,
        input_shape_pkcast,
        output_shape_pkcast,
    )

    with torch.no_grad():
        x_cond_norm = cfm_model.normalize(x_cond_latent).permute(0, 2, 3, 4, 1)
        bsz, _, hz, wz, cz = x_cond_norm.shape
        t_future = x_true.shape[2]
        sample_predictions = []

        for sample_idx in range(args.probabilistic_samples):
            torch.manual_seed(args.seed + sample_idx)
            x0_noise = torch.randn((bsz, t_future, hz, wz, cz), device=device)
            x0_flat = x0_noise.view(bsz * t_future, hz, wz, cz)

            def flow_dynamics(t, x_flat):
                x_flow_local = x_flat.view(bsz, t_future, hz, wz, cz)
                t_batched = t * torch.ones(bsz, device=x_flow_local.device)
                v_t = cfm_model(t_batched, x_flow_local, x_cond_norm)
                return v_t.view(bsz * t_future, hz, wz, cz)

            t_span = torch.tensor([0.0, 1.0], device=device)
            if euler_steps == 0:
                solution = odeint(
                    flow_dynamics,
                    x0_flat,
                    t_span,
                    method="adaptive_heun",
                    rtol=1e-2,
                    atol=1e-3,
                    adjoint_params=cfm_model.parameters(),
                )
            else:
                solution = odeint(
                    flow_dynamics,
                    x0_flat,
                    t_span,
                    method="euler",
                    options={"step_size": 1.0 / float(euler_steps)},
                    rtol=1e-2,
                    atol=1e-3,
                    adjoint_params=cfm_model.parameters(),
                )

            x_pred_sample = solution[-1].view(bsz, t_future, hz, wz, cz)
            sample_predictions.append(x_pred_sample.unsqueeze(1))

        x_pred = torch.cat(sample_predictions, dim=1)
        x_pred = x_pred * cfm_model.std + cfm_model.mean

        bsz, samples, timesteps, h_lat, w_lat, c_lat = x_pred.shape
        x_pred = x_pred.reshape(bsz * samples * timesteps, h_lat, w_lat, c_lat)
        x_pred = x_pred.permute(0, 3, 1, 2)

        decoded_chunks = []
        decode_batch = batch_size_autoencoder or x_pred.shape[0]
        for start in range(0, x_pred.shape[0], decode_batch):
            decoded_obj = safe_decode(vae_model, x_pred[start : start + decode_batch])
            decoded_chunks.append(decoded_obj.sample)
        x_pred = torch.cat(decoded_chunks, dim=0)

        if normalized_autoencoder:
            x_pred = x_pred * 255.0

    x_pred = x_pred.reshape(
        bsz, samples, timesteps, x_pred.shape[1], x_pred.shape[2], x_pred.shape[3]
    )
    x_pred = x_pred.permute(0, 1, 2, 4, 5, 3)
    if x_pred.shape[-1] == 1:
        x_pred = x_pred.squeeze(-1)

    x_pred_np = post_process_samples(
        x_pred.cpu().numpy().astype(np.float32), clamp_min=0.0, clamp_max=255.0
    )
    x_cond_np = x_cond.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    x_true_np = x_true.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    prediction_frames = x_pred_np[0, 0]
    input_frames = post_process_samples(x_cond_np, clamp_min=0.0, clamp_max=255.0)
    target_frames = post_process_samples(x_true_np, clamp_min=0.0, clamp_max=255.0)

    prediction_gif = os.path.join(
        args.output_dir, f"sample_{args.sample_index:05d}_prediction.gif"
    )
    input_gif = os.path.join(
        args.output_dir, f"sample_{args.sample_index:05d}_input.gif"
    )
    target_gif = os.path.join(
        args.output_dir, f"sample_{args.sample_index:05d}_target.gif"
    )
    npz_path = os.path.join(
        args.output_dir, f"sample_{args.sample_index:05d}_prediction.npz"
    )

    save_gif(
        input_frames,
        metadata,
        input_gif,
        title="Input Sequence",
        fps=args.fps,
        cartopy_features=args.cartopy_features,
    )
    save_gif(
        target_frames,
        metadata,
        target_gif,
        title="Target Sequence",
        fps=args.fps,
        cartopy_features=args.cartopy_features,
    )
    save_gif(
        prediction_frames,
        metadata,
        prediction_gif,
        title="Predicted Sequence",
        fps=args.fps,
        cartopy_features=args.cartopy_features,
    )

    np.savez_compressed(
        npz_path,
        input=input_frames,
        target=target_frames,
        prediction=prediction_frames,
    )

    print(f"Saved input GIF: {input_gif}")
    print(f"Saved target GIF: {target_gif}")
    print(f"Saved prediction GIF: {prediction_gif}")
    print(f"Saved arrays: {npz_path}")


if __name__ == "__main__":
    main()
