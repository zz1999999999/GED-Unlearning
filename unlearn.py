import os
import click
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips

import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from PIL import Image
from arcface import IDLoss
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import pandas as pd


def save_loss_record(loss_record, file_path="loss_record.csv"):
    max_len = max(len(v) for v in loss_record.values())

    for k, v in loss_record.items():
        if len(v) < max_len:
            v.extend([np.nan] * (max_len - len(v))) 
    df = pd.DataFrame(loss_record)
    df.to_csv(file_path, index=False)
    print(f"Loss record saved to: {file_path}")

def tensor_to_image(t):
    t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(t[0].cpu().numpy(), "RGB")

def image_to_tensor(i, size=256):
    i = i.resize((size, size))
    i = np.array(i)
    i = i.transpose(2, 0, 1)
    i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
    return i
def compute_local_div_loss(lpips_fn, img_u, img_target, T=2.0):
    lpips_value = lpips_fn(img_u, img_target)
    return torch.exp(-T * lpips_value).mean()
# def compute_local_div_loss(lpips_fn, img_u, img_target, T=5.0):
#     lpips_value = lpips_fn(img_u, img_target) 
#     lpips_value = lpips_value.view(lpips_value.size(0), -1)  
#     lpips_value = torch.clamp(lpips_value, min=1e-4) 
#     return (1.0 / (1.0 + T * lpips_value)).mean()
def compute_local_loss0(mse, temperature=0.5):
    loss_local_div0 = torch.exp(-mse / temperature)
    return loss_local_div0
def generate_distant_w(z_rg, rho=1.0, max_B=3.0):
    rand_num = torch.empty(1).uniform_(-max_B, max_B).item()
    if rand_num < 0:
        B = -rand_num
    else:
        B = rand_num
    perturb  = torch.empty_like(z_rg).uniform_(-B, B)
    z_ne = z_rg + perturb + torch.sign(perturb) * rho
    return z_ne
intermediate_outputs = {}
hooks = []

def get_hook(name, output_dict):
    def hook(module, input, output):
        output_dict[name] = output.detach().cpu()
    return hook

def register_hooks(model, output_dict):
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(get_hook(name, output_dict))
            hooks.append(h)

def remove_hooks():
    for h in hooks:
        h.remove()
    hooks.clear()
def generate_reasonable_w_from_original(w_rg, w_u, max_scale=10.0, min_norm=1.0, loss_adj_alpha_range_max=1.0):
    delta = w_rg - w_u
    norm_delta = delta.norm(p=2)
    if norm_delta < min_norm:
        offset = loss_adj_alpha_range_max * delta / (norm_delta+ 1e-6)  # Avoid division by zero
    else:
        scale_factor = 1.0 + (max_scale - 1.0) * torch.tanh(1.0 / (norm_delta + 1e-6))
        offset = delta * scale_factor
    return w_u + offset


class GumbelMaskNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, temperature=1e-4):
        super(GumbelMaskNet, self).__init__()
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output logits
        )

    def forward(self, x):
        x = x.float()  
        B, N, D = x.shape
        x_flat = x.view(B * N, D)                 # [B*N, D]
        logits = self.mlp(x_flat)                 # [B*N, D]

        # Sample Gumbel noise
        noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(noise))  # standard Gumbel(0, 1)

        y = (logits + gumbel_noise) / self.temperature  # [B*N, D]
        mask = torch.sigmoid(y)                         # [B*N, D]

        return mask.view(B, N, D) 

def mutual_info_loss(masked_A, pos_sample, neg_sample, temperature=0.07):
    """
    masked_A:     [B, C, H, W]
    pos_sample:   [B, C, H, W]
    neg_sample:   [B, C, H, W] or [B, N_neg, C, H, W]
    """
    B, C, H, W = masked_A.shape

    # Normalize on channel dimension
    masked_A = F.normalize(masked_A, dim=1)     # [B, C, H, W]
    pos_sample = F.normalize(pos_sample, dim=1) # [B, C, H, W]
    neg_sample = F.normalize(neg_sample, dim=1) # [B, C, H, W]

    # Flatten spatial dimensions: [B, C, H*W]
    masked_A_flat = masked_A.view(B, C, -1)     # [B, C, N]
    pos_flat = pos_sample.view(B, C, -1)        # [B, C, N]
    neg_flat = neg_sample.view(B, C, -1)        # [B, C, N]

    # Dot product similarity at each spatial location
    pos_sim = (masked_A_flat * pos_flat).sum(dim=1) / temperature  # [B, N]
    neg_sim = (masked_A_flat * neg_flat).sum(dim=1) / temperature  # [B, N]

    # Combine positive and negative
    logits = torch.stack([pos_sim, neg_sim], dim=1)  # [B, 2, N]
    log_probs = F.log_softmax(logits, dim=1)         # [B, 2, N]

    # Take negative log-likelihood of positive
    loss = -log_probs[:, 0, :].mean()                # average over batch and pixels

    return loss


def create_loss_record(local=True, adj=True, global_=True, globa_extra=True ,mask_use_flag=True, eval_similar=True, use_filter=True, filter_layer=True):
    loss_record = {}

    if local:
        loss_record.update({
            "loss_local_mse": [],
            "loss_local_lpips": [],
            "loss_local_id": []
        })

    if adj:
        loss_record.update({
            "loss_adj_mse": [],
            "loss_adj_lpips": [],
            "loss_adj_id": []
        })

    if global_:
        if globa_extra:
            loss_record.update({
                "loss_glob_mse": [],
                "loss_glob_lpips": [],
                "loss_glob_id": []
            })
        else:
            loss_record.update({
                "loss_glob_lpips": [],
            })

    if mask_use_flag:
        loss_record["mask_use"] = []
        loss_record["mask_sparsity"] = []
    if eval_similar:
        loss_record["retain_s"] = []
        loss_record["forget_s"] = []
        loss_record["retain_c"] = []
        loss_record["forget_c"] = []
        loss_record["sim_to_target"] = []
        loss_record["latent_shift"] = []
        # loss_record["hessian_trace"] = []
    if use_filter:
        if filter_layer:
            loss_record["layers"] = []
        else:
            loss_record["nums"] = []

    return loss_record



@click.command()
@click.option("--pretrained_ckpt", type=str, default="ffhqrebalanced512-128.pkl")
@click.option("--iter", type=int, default=1000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--seed", type=int, default=None)
@click.option("--fov-deg", type=float, default=18.837)
@click.option("--truncation_psi", type=float, default=1.0)
@click.option("--truncation_cutoff", type=int, default=14)
@click.option("--exp", type=str, required=True)
@click.option("--inversion", type=str, default=None)
@click.option("--inversion_image_path", type=str, default=None)
@click.option("--angle_p", type=float, default=-0.2)
@click.option("--angle_y_abs", type=float, default=np.pi / 12)
@click.option("--sample_views", type=int, default=11)
# latent target unlearning: local unlearning loss
@click.option("--local", is_flag=True)
@click.option("--loss_local_mse_lambda", type=float, default=1e-2)
@click.option("--loss_local_lpips_lambda", type=float, default=1.0)
@click.option("--loss_local_id_lambda", type=float, default=0.1)
# latent target unlearning: adjacency-aware unlearning loss
@click.option("--adj", is_flag=True)
@click.option("--loss_adj_mse_lambda", type=float, default=1e-2)
@click.option("--loss_adj_lpips_lambda", type=float, default=1.0)
@click.option("--loss_adj_id_lambda", type=float, default=0.1)
@click.option("--loss_adj_batch", type=int, default=1)
@click.option("--loss_adj_lambda", type=float, default=1.0)
@click.option("--loss_adj_alpha_range_min", type=int, default=0)
@click.option("--loss_adj_alpha_range_max", type=int, default=15)
# latent target unlearning: global preservation loss
@click.option("--glob", is_flag=True)
@click.option("--loss_global_lambda", type=float, default=1.0)
@click.option("--loss_global_batch", type=int, default=1)
@click.option("--loss_global_mse_sim_lambda", type=float, default=1.0)
@click.option("--loss_global_cos_sim_lambda", type=float, default=10.0)
@click.option("--loss_global_orth_lambda", type=float, default=1.0)
# latent target unlearning: un-identifying face on latent space
@click.option("--target_idx", type=int, default=0)
@click.option("--target", type=str, default="extra")
@click.option("--target_d", type=float, default=30.0)
# latent target unlearning: target-free unlearning loss
# @click.option("--target_free", is_flag=False, default=False)
# @click.option("--loss_discriminator_lambda", type=float, default=0.5)
# @click.option("--loss_target_free_lambda", type=float, default=10.0)
# @click.option("--loss_retain_lpips_lambda", type=float, default=1.0)
# @click.option("--loss_retain_id_lambda", type=float, default=1.0)
@click.option("--mask", default=False)
@click.option("--loss_glob_mask_lambda", type=float, default=1.0)
@click.option("--loss_glob_mse_lambda", type=float, default=0.001)
@click.option("--loss_sparsity", type=float, default=0.5)
@click.option("--threshold", type=float, default=0.1)


@click.option("--use_filter", default=False)
@click.option("--orthogonal",default=False)
@click.option("--globa_extra", default=False)
@click.option("--filter_start", type=int, default=0)
@click.option("--filter_end", type=int, default=1000)
@click.option("--filter_layer", default=False)

@click.option("--other_methods", type=str, default=None)
@click.option("--other_methods_lambda", type=float, default=1)

def unlearn(*args, **kwargs):
    pretrained_ckpt = kwargs["pretrained_ckpt"]
    pretrained_ckpt = os.path.join("/root/project/", pretrained_ckpt)
    iter = kwargs["iter"]
    lr = kwargs["lr"]
    seed = kwargs["seed"]
    fov_deg = kwargs["fov_deg"]
    truncation_psi = kwargs["truncation_psi"]
    truncation_cutoff = kwargs["truncation_cutoff"]
    exp = kwargs["exp"]
    inversion = kwargs["inversion"]
    inversion_image_path = kwargs["inversion_image_path"]

    angle_p = kwargs["angle_p"]
    angle_y_abs = kwargs["angle_y_abs"]
    sample_views = kwargs["sample_views"]

    local = kwargs["local"]
    loss_local_mse_lambda = kwargs["loss_local_mse_lambda"]
    loss_local_lpips_lambda = kwargs["loss_local_lpips_lambda"]
    loss_local_id_lambda = kwargs["loss_local_id_lambda"]

    adj = kwargs["adj"]
    loss_adj_mse_lambda = kwargs["loss_adj_mse_lambda"]
    loss_adj_lpips_lambda = kwargs["loss_adj_lpips_lambda"]
    loss_adj_id_lambda = kwargs["loss_adj_id_lambda"]
    loss_adj_batch = kwargs["loss_adj_batch"]
    loss_adj_lambda = kwargs["loss_adj_lambda"]
    loss_adj_alpha_range_min = kwargs["loss_adj_alpha_range_min"]
    loss_adj_alpha_range_max = kwargs["loss_adj_alpha_range_max"]

    glob = kwargs["glob"]
    loss_global_lambda = kwargs["loss_global_lambda"]
    loss_global_batch = kwargs["loss_global_batch"]
    
    # target_free = kwargs["target_free"]
    # loss_target_free_lambda = kwargs["loss_target_free_lambda"]
    # loss_discriminator_lambda = kwargs["loss_discriminator_lambda"]

    target_idx = kwargs["target_idx"]
    target = kwargs["target"]
    target_d = kwargs["target_d"]
    mask_use = kwargs["mask"]
    orthogonal = kwargs["orthogonal"]

    # loss_retain_lpips_lambda = kwargs["loss_retain_lpips_lambda"]
    # loss_retain_id_lambda = kwargs["loss_retain_id_lambda"]
    loss_glob_mask_lambda = kwargs["loss_glob_mask_lambda"]
    loss_glob_mse_lambda = kwargs["loss_glob_mse_lambda"]
    loss_sparsity = kwargs["loss_sparsity"]

    use_filter = kwargs["use_filter"]
    globa_extra = kwargs["globa_extra"]
    eval_similar = True
    record_output = False
    filter_start = kwargs["filter_start"]
    filter_end = kwargs["filter_end"]
    threshold = kwargs["threshold"]
    filter_layer = kwargs["filter_layer"]
    loss_global_mse_sim_lambda =kwargs["loss_global_mse_sim_lambda"]
    loss_global_cos_sim_lambda = kwargs["loss_global_cos_sim_lambda"]
    loss_global_orth_lambda = kwargs["loss_global_orth_lambda"]
    use_RMU = False
    use_gradient_ascent = False
    use_gmu = False
    PCGrad = False

    if kwargs["other_methods"] is not None:
        if kwargs["other_methods"] == "RMU":
            use_RMU = True
        elif kwargs["other_methods"] == "gradient_ascent":
            use_gradient_ascent = True
        elif kwargs["other_methods"] == "gmu":
            use_gmu = True
        elif kwargs["other_methods"] == "pcgrad":
            PCGrad = True

    other_methods_lambda = kwargs["other_methods_lambda"]
    if use_filter:
        print("Use filter")
        print(f"Start_index-{filter_start}///End_index-{filter_end}")
        if filter_layer:
            print("Use filter Layer")
        else:
            print("Use filter Node")
    if eval_similar:
        print("Start evaluate")
    if globa_extra:
        print("Extra global loss")


    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    print(exp)
    if local:
        print("Local unlearning loss enabled.")
    if adj:
        print("Adjacency-aware unlearning loss enabled.")
    if glob:
        print("Global preservation loss enabled.")
    if mask_use:
        print("Mask net enabled.")
    

    device = torch.device("cuda")
    with dnnlib.util.open_url(pretrained_ckpt) as f:
        data = legacy.load_network_pkl(f)
        g_source = data["G_ema"].to(device)
        # if use_RMU:
        #     discriminator = data["D"].to(device)
        #     discriminator.eval()
        #     for p in discriminator.parameters():
        #         p.requires_grad = False
    print(f"address_{pretrained_ckpt}")


    mtcnn = MTCNN(image_size=160, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def tensor_to_pil(image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            image_tensor = image_tensor.cpu().clamp(0, 1)
            return transforms.ToPILImage()(image_tensor)
        return image_tensor  

    def compare_faces_tensor(img1_tensor, img2_tensor, device=device):
        img1 = tensor_to_pil(img1_tensor)
        img2 = tensor_to_pil(img2_tensor)

        face1 = mtcnn(img1)
        face2 = mtcnn(img2)

        if face1 is None or face2 is None:
            raise ValueError("At least one image has no detected faces")

        emb1 = model(face1.unsqueeze(0).to(device))
        emb2 = model(face2.unsqueeze(0).to(device))
        sim = F.cosine_similarity(emb1, emb2).item()
        return sim

    
    generator = TriPlaneGenerator(*g_source.init_args, **g_source.init_kwargs).requires_grad_(False).to(device)
    copy_params_and_buffers(g_source, generator, require_all=True)
    generator.neural_rendering_resolution = g_source.neural_rendering_resolution
    generator.rendering_kwargs = g_source.rendering_kwargs
    generator.load_state_dict(g_source.state_dict(), strict=False)
    generator.train()

    g_source = copy.deepcopy(generator)
    # if use_RMU:
    # #     discriminator.eval()
    # #     for p in discriminator.parameters():
    # #         p.requires_grad = False
    for name, param in g_source.named_parameters():
        param.requires_grad = False

    for name, param in generator.named_parameters():
        if "backbone.synthesis" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    
    exp_dir = f"experiments/{exp}"
    ckpt_dir = f"experiments/{exp}/checkpoints"
    image_dir = f"experiments/{exp}/training/images"
    result_dir = f"experiments/{exp}/training/results"
    output_dir = f"experiments/{exp}/training/outputs"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)



    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg in kwargs:
            f.write(f"{arg}: {kwargs[arg]}\n")

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
    cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device)
    camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)

    w_avg = torch.load("/root/project/w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(0) # [1, 14, 512]
    if mask_use:
        mask_net = GumbelMaskNet().to(device)
        mask_optimizer = optim.Adam(mask_net.parameters(), lr=lr)

        mask_net.train()


    # Visualize before unlearning
    with torch.no_grad():
        if inversion is not None:
            assert inversion_image_path is not None, "The path of an image to invert is required."
            assert inversion in ["goae"]
            if inversion == "goae":
                from goae import GOAEncoder
                from goae.swin_config import get_config
                
                swin_config = get_config()
                stage_list = [10000, 20000, 30000]
                encoder_ckpt = "/root/project/encoder_FFHQ.pt"

                encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(device)
                encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
                if os.path.isdir(inversion_image_path):
                    filenames = sorted(os.listdir(inversion_image_path))
                    imgs = [image_to_tensor(Image.open(os.path.join(inversion_image_path, filename)).convert("RGB")) for filename in filenames]   
                    imgs = torch.stack(imgs, dim=0)
                    imgs = imgs.to(device)
                    w, _ = encoder(imgs)
                    w_origin = w + w_avg
                    w_u = w[[target_idx], :, :] + w_avg
                    w_retain = w[[-2], :, :] + w_avg
                    del imgs
                else:
                    img = image_to_tensor(Image.open(inversion_image_path).convert("RGB")).unsqueeze(0)
                    w, _ = encoder(img)
                    w_u = w + w_avg
                    del img
            else:
                raise NotImplementedError
        else:
            w_avg = torch.load("/root/project/w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(0) # [1, 14, 512]
            if inversion_image_path is not None:
                w_u = torch.load(inversion_image_path)
            else:
                z_u = torch.randn(1, 512, device=device)
                w_u = generator.mapping(z_u, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        generator.eval()
        if inversion is None: # for random
            for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                img_u = generator.synthesis(w_u, camera_params_view)["image"]
                img_u = tensor_to_image(img_u)
                img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
            del img_u
        else:
            if os.path.isdir(inversion_image_path): # for OOD
                for i in range(len(filenames)):
                    for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                        camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                del img_origin
            else: # for InD
                for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
        generator.train()
    
    if target == "average":
        w_target = w_avg
    elif target == "extra":
        with torch.no_grad():
            if inversion is not None:
                w_id = w[[target_idx], :, :]
            else:
                w_id = w_u - w_avg
            w_target = w_avg - w_id / w_id.norm(p=2) * target_d
    

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    id_fn = IDLoss().to(device)

    pbar = tqdm(range(iter))
    loss_record = create_loss_record(local=local, adj=adj, global_=glob, globa_extra=globa_extra, mask_use_flag=mask_use, eval_similar=eval_similar, use_filter=use_filter, filter_layer=filter_layer)
    for i in pbar:
        angle_y = np.random.uniform(-angle_y_abs, angle_y_abs)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        loss_retain = torch.tensor(0.0, device=device)
        loss_forget = torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        if mask_use:
            loss_mask = torch.tensor(0.0, device=device)
        loss_dict = {}

        # local unlearning loss
        if local:
            loss_local = torch.tensor(0.0, device=device)
            if mask_use:
                mask_u = mask_net(w_u)
                w_u_masked = w_u * mask_u
                mask_target = mask_net(w_target)
                w_target_mask = w_target * mask_target
            else:
                w_u_masked = w_u
                w_target_mask = w_target 
            feat_u = generator.get_planes(w_u)
            feat_target = g_source.get_planes(w_target)
            loss_local_mse = F.mse_loss(feat_u, feat_target)
            loss_local = loss_local + loss_local_mse_lambda * loss_local_mse

            img_u = generator.synthesis(w_u_masked, camera_params)["image"]
            img_target = g_source.synthesis(w_target_mask, camera_params)["image"]
            loss_local_lpips = lpips_fn(img_u, img_target).mean()
            loss_local = loss_local + loss_local_lpips_lambda * loss_local_lpips

            loss_local_id = id_fn(img_u, img_target)
            loss_local = loss_local + loss_local_id_lambda * loss_local_id

            # loss = loss + loss_local
            loss_forget = loss_forget + loss_local
            loss_dict["loss_local"] = loss_local.item()
            loss_record["loss_local_mse"].append(loss_local_mse.item())
            loss_record["loss_local_lpips"].append(loss_local_lpips.item())
            loss_record["loss_local_id"].append(loss_local_id.item())
            del img_u, img_target

        # adjacency-aware unlearning loss
        if adj:
            loss_adj = torch.tensor(0.0, device=device)
            for _ in range(loss_adj_batch):
                z_ra = torch.randn(1, 512, device=device)
                w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

                if loss_adj_alpha_range_max is not None:
                    loss_adj_alpha = torch.from_numpy(np.random.uniform(loss_adj_alpha_range_min, loss_adj_alpha_range_max, size=1)).unsqueeze(1).unsqueeze(1).to(device)
                deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                w_u_adj = w_u + deltas
                w_target_adj = w_target + deltas
                if mask_use:
                    w_ng = 50* (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    mask_adj = mask_net(w_u_adj)
                    w_u_adj_mask = w_u_adj * mask_adj
                    mask_target = mask_net(w_target_adj)
                    w_target_adj_mask = w_target_adj * mask_target
                else:
                    w_u_adj_mask = w_u_adj
                    w_target_adj_mask = w_target_adj
                

                feat_u = generator.get_planes(w_u_adj)
                feat_target = g_source.get_planes(w_target_adj)

                loss_adj_mse = F.mse_loss(feat_u, feat_target)
                loss_adj = loss_adj + loss_adj_mse_lambda * loss_adj_mse
            
                img_u = generator.synthesis(w_u_adj_mask, camera_params)["image"]
                
                img_target = g_source.synthesis(w_target_adj_mask, camera_params)["image"]
                loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                
                loss_adj = loss_adj + loss_adj_lpips_lambda * loss_adj_lpips

                loss_adj_id = id_fn(img_u, img_target)
                loss_adj = loss_adj + loss_adj_id_lambda * loss_adj_id 

            # loss = loss + loss_adj_lambda * loss_adj
            loss_forget = loss_forget + loss_adj_lambda * loss_adj
            loss_dict["loss_adj"] = loss_adj.item()
            loss_record["loss_adj_mse"].append(loss_adj_mse.item())
            loss_record["loss_adj_lpips"].append(loss_adj_lpips.item())
            loss_record["loss_adj_id"].append(loss_adj_id.item())
            del img_u, img_target
            

        # global preservation loss
        if glob:
            loss_global = torch.tensor(0.0, device=device)
            for _ in range(loss_global_batch):
                z_rg = torch.randn(1, 512, device=device)
                w_rg = generator.mapping(z_rg, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                # w_rg = generate_reasonable_w_from_original(w_rg, w_u, max_scale=3*loss_adj_alpha_range_max, min_norm=1.0)
                if globa_extra:
                    feat_u = generator.get_planes(w_rg)
                    feat_target = g_source.get_planes(w_rg)
                    loss_glob_mse = F.mse_loss(feat_u, feat_target)
                img_u = generator.synthesis(w_rg, camera_params)["image"]
                img_target = g_source.synthesis(w_rg, camera_params)["image"]
                img_f_like = generator.synthesis(w_u, camera_params)["image"]
                img_u0 = tensor_to_image(img_u)
                img_u0 = image_to_tensor(img_u0).unsqueeze(0).to(device)
                img_u1 = tensor_to_image(img_target)
                img_u1 = image_to_tensor(img_u1).unsqueeze(0).to(device)
                w0, _ = encoder(img_u0)
                w1, _ = encoder(img_u1)
                cos_sim = F.cosine_similarity(w0, w1, dim=1).mean()
                loss_global_lpips = lpips_fn(img_u, img_target)
                loss_global_cos_sim = 1 - cos_sim
                loss_global_mse_sim = F.mse_loss(w0, w1)
                # if torch.norm(w_rg-w_u, p=2) > loss_adj_alpha_range_max and torch.norm(w_rg-w_target, p=2) > loss_adj_alpha_range_max:
                cos_sim_target =F.cosine_similarity(w0, w_target, dim=1).mean()
                eps = 1e-6
                mse_loss = loss_global_mse_sim/(F.mse_loss(w0, w_target) + eps)
                loss_global_cos_sim = loss_global_cos_sim + torch.sigmoid(cos_sim_target) + mse_loss
                if orthogonal:
                    lpips_pos = lpips_fn(img_u, img_target).mean() 
                    lpips_neg = lpips_fn(img_u, img_f_like).mean()
                    margin = 0.3  
                    loss_retain_disentangle = F.relu(lpips_neg - lpips_pos + margin)
                    cos_sim_latent = F.cosine_similarity(w0, w_u, dim=1).mean()
                    loss_latent_orth = F.relu(cos_sim_latent - margin)
                    color_mean_u = img_u.mean(dim=[2,3])
                    color_mean_f = img_f_like.mean(dim=[2,3])
                    loss_color_shift = F.mse_loss(color_mean_u, color_mean_f.detach())

                if globa_extra:
                    loss_glob_id = id_fn(img_u, img_target)
                    loss_global = loss_global + loss_global_lpips + \
                                loss_glob_mse_lambda *loss_glob_mse +  \
                                loss_glob_id + \
                                loss_global_mse_sim_lambda * loss_global_mse_sim + \
                                loss_global_cos_sim_lambda * loss_global_cos_sim 
                    if orthogonal:
                        loss_global = loss_global + loss_global_orth_lambda * (loss_latent_orth + loss_retain_disentangle + loss_color_shift)
                else:
                    loss_global = loss_global + \
                                loss_global_lpips + \
                                loss_global_mse_sim_lambda * loss_global_mse_sim + \
                                loss_global_cos_sim_lambda * loss_global_cos_sim 
                    if orthogonal:
                        loss_global = loss_global + loss_global_orth_lambda * (loss_latent_orth + loss_retain_disentangle + loss_color_shift)
            # loss = loss + loss_global_lambda * loss_global
            loss_retain = loss_retain + loss_global_lambda * loss_global
            loss_dict["loss_global"] = loss_global.item()
            loss_record["loss_glob_lpips"].append(loss_global_lpips.item())
            if globa_extra:
                loss_record["loss_glob_mse"].append(loss_glob_mse.item())
                loss_record["loss_glob_id"].append(loss_glob_id.item())
            del img_u, img_target, img_f_like
            
        if use_gradient_ascent:
            feat_u = generator.get_planes(w_u)
            feat_target = g_source.get_planes(w_u)
            loss_anti_id = F.cosine_similarity(feat_u, feat_target).mean()
            del feat_u, feat_target

        if use_RMU:
            z = torch.randn(1, 512, device=device)
            w = g_source.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img_u_forget = generator.synthesis(w_u, camera_params)["image"]
            img_random = g_source.synthesis(w, camera_params)["image"]
            loss_RMU = lpips_fn(img_u_forget, img_random).mean()
            # with torch.no_grad():
            #     imgs_for_D = {'image': img_u_forget, 'image_raw': img_u_forget}
            #     d_logit = discriminator(imgs_for_D, camera_params)
            # loss_RMU = F.softplus(d_logit).mean()
            del img_u_forget, img_random
        if use_gmu:
            img_u = generator.synthesis(w_u, camera_params)["image"]
            img_u_original = g_source.synthesis(w_u, camera_params)["image"]
            loss_for_gmu = -lpips_fn(img_u, img_u_original).mean()
            trainable_params = [p for p in generator.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss_for_gmu, trainable_params, create_graph=False)
            loss_gmu_penalty = sum(g.norm(p=2).pow(2) for g in grads)


            del img_u, img_u_original

            # img_u_mask = generator.synthesis(w_u*mask_u, camera_params)["image"]
            # img_u = generator.synthesis(w_u, camera_params)["image"]
            # imgs = {'image': img_u,'image_raw': img_u_mask}
            # d_score = discriminator(imgs, camera_params).mean()
            # loss = loss + loss_target_free_lambda * loss_anti_id + loss_discriminator_lambda * (-d_score)
        # discriminator loss
        if mask_use:
            z_rg = torch.randn(1, 512, device=device)
            z_ne = generate_distant_w(z_rg, rho=40.0, max_B=5.0)
            w_rg = g_source.mapping(z_rg, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            w_ne = g_source.mapping(z_ne, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

            mask_prob = mask_net(w_rg)
            w_rg_masked = w_rg * mask_prob


            img_rg_masked = g_source.synthesis(w_rg_masked, camera_params)["image"]
            img_rg = g_source.synthesis(w_rg, camera_params)["image"]
            img_ne = g_source.synthesis(w_ne, camera_params)["image"]
            img_inverse_masked = g_source.synthesis(w_rg*(1-mask_prob), camera_params)["image"]

            loss_mask_id_exclude = torch.relu(id_fn(img_rg_masked, img_rg) - id_fn(img_inverse_masked, img_rg)+0.5).mean()
            loss_mask_lpips_exclude = torch.relu(lpips_fn(img_rg_masked, img_rg) - lpips_fn(img_inverse_masked, img_rg) + 0.5).mean()
            mask_sparsity = mask_prob.abs().mean()
            loss_mask = loss_mask + (mutual_info_loss(img_rg_masked, img_rg, img_ne) + loss_mask_id_exclude + loss_mask_lpips_exclude)* loss_glob_mask_lambda + loss_sparsity * mask_sparsity
            loss_record["mask_use"].append(loss_mask.item())
            loss_record["mask_sparsity"].append(loss_sparsity)
            del img_rg_masked,img_rg,img_ne, img_inverse_masked
        pbar.set_description(f"Loss: {loss.item():.4f}, Retain: {loss_retain.item():.4f}, Forget: {loss_forget.item():.4f}, Mask: {loss_mask.item() if mask_use else 0:.4f}")
        if eval_similar:
            with torch.no_grad():
                generator.eval()

                imgs_n, latents_n, sim_to_target_list = [], [], []
                retain_s, retain_c, counts, countc = 0, 0, 0, 0
                while counts<10:
                    z = torch.randn(1, 512, device=device)
                    w = generator.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                    img0 = generator.synthesis(w, camera_params)["image"]
                    img1 = g_source.synthesis(w, camera_params)["image"]
                    
                    try:
                        sim_r = compare_faces_tensor(img0, img1)
                        retain_s += sim_r
                        counts += 1
                    except ValueError as e:
                        continue
                    img_u0 = tensor_to_image(img0)
                    img_u0 = image_to_tensor(img_u0).unsqueeze(0).to(device)
                    img_u1 = tensor_to_image(img1)
                    img_u1 = image_to_tensor(img_u1).unsqueeze(0).to(device)
                    w0, _ = encoder(img_u0)
                    w1, _ = encoder(img_u1)
                    cos_sim_layerwise = F.cosine_similarity(w0, w1, dim=1)
                    cos_sim = cos_sim_layerwise.mean().item()
                    retain_c += cos_sim
                    countc += 1

                    imgs_n.append(img_u0)
                    latents_n.append(w0.detach())
                    sim_to_target = F.cosine_similarity(w0, w_target, dim=1).mean().item()
                    sim_to_target_list.append(sim_to_target)


                    del img0, img1, img_u0, img_u1, w0, w1                   
                torch.cuda.empty_cache()
                loss_record["retain_c"].append(retain_c/countc)
                loss_record["retain_s"].append(retain_s/counts)
                loss_record["sim_to_target"].append(np.mean(sim_to_target_list))
                if len(latents_n) > 1:
                    latents_n_stack = torch.cat(latents_n, dim=0)  # [N, 512]
                    latent_mean = latents_n_stack.mean(dim=0).unsqueeze(0)
                    latent_shift = F.mse_loss(latent_mean, w_target).item()
                    loss_record["latent_shift"].append(latent_shift)

                img3 = generator.synthesis(w_u, camera_params)["image"]
                img4 = generator.synthesis(w_target, camera_params)["image"]
                try:
                    sim_f = compare_faces_tensor(img3, img4)
                    loss_record["forget_s"].append(sim_f)
                except ValueError as e:
                    loss_record["forget_s"].append(-1)
                img_u3 = tensor_to_image(img3)
                img_u3 = image_to_tensor(img_u3).unsqueeze(0).to(device)
                img_u4 = tensor_to_image(img4)
                img_u4 = image_to_tensor(img_u4).unsqueeze(0).to(device)
                w3, _ = encoder(img_u3)
                w4, _ = encoder(img_u4)
                cos_sim_layerwise_f = F.cosine_similarity(w3, w4, dim=1)
                cos_sim_f = cos_sim_layerwise_f.mean().item()
                loss_record["forget_c"].append(cos_sim_f)
                del img3, img4, img_u3, img_u4, w3, w4
                generator.train()
                
        if use_filter and filter_start <= i <= filter_end:
            optimizer_G.zero_grad()
            named_trainable_params = [(name, p) for name, p in generator.named_parameters() if p.requires_grad]
            param_names = [name for name, _ in named_trainable_params]
            trainable_params = [p for _, p in named_trainable_params]
            retain_grads = torch.autograd.grad(loss_retain, trainable_params, create_graph=False, retain_graph=True)
            forget_grads = torch.autograd.grad(loss_forget, trainable_params, create_graph=False, retain_graph=True)
            filtered_grads = []
            updated_layers = []
            if filter_layer:
                for name, gr, gf in zip(param_names, retain_grads, forget_grads):
                    if torch.norm(gr) == 0 or torch.norm(gf) == 0:
                        sim = torch.tensor(0.0, device=gr.device) 
                    else:
                        sim = F.cosine_similarity(gr.reshape(-1), gf.reshape(-1), dim=0)
                    if sim > threshold:
                        filtered_grads.append(gf)       
                    else:
                        filtered_grads.append(torch.zeros_like(gf))
                    updated_layers.append((name, sim.item()))
                with torch.no_grad():
                    for p, g in zip(trainable_params, filtered_grads):
                        p.grad = g
                
                loss_record["layers"].append(updated_layers)
            else:
                for name, gr, gf in zip(param_names, retain_grads, forget_grads):
                    epsilon = 1e-8
                    mask = (gr * gf > epsilon).float()  
                    filtered_grad = gf * mask 
                    filtered_grads.append(filtered_grad)

                    num_updated = mask.sum().item()
                    total = mask.numel()
                    updated_layers.append((name, num_updated/total))

                with torch.no_grad():
                    for p, g in zip(trainable_params, filtered_grads):
                        p.grad = g
                loss_record["nums"].append(updated_layers)
            optimizer_G.step()
            if mask_use:
                mask_optimizer.zero_grad()
                loss_mask.backward()
                mask_optimizer.step()
        elif PCGrad:
            optimizer_G.zero_grad()
            named_trainable_params = [(name, p) for name, p in generator.named_parameters() if p.requires_grad]
            param_names = [name for name, _ in named_trainable_params]
            trainable_params = [p for _, p in named_trainable_params]
            retain_grads = torch.autograd.grad(loss_retain, trainable_params, create_graph=False, retain_graph=True)
            forget_grads = torch.autograd.grad(loss_forget, trainable_params, create_graph=False, retain_graph=True)
            def flatten_grads(grads):
                return torch.cat([g.reshape(-1) for g in grads])
            def unflatten_grads(flat_grad, grads_like):
                grads = []
                idx = 0
                for g in grads_like:
                    numel = g.numel()
                    grads.append(flat_grad[idx:idx+numel].reshape_as(g))
                    idx += numel
                return grads
            retain_vec = flatten_grads(retain_grads)
            forget_vec = flatten_grads(forget_grads)
            proj_retain_on_forget = (retain_vec @ forget_vec) / (forget_vec @ forget_vec + 1e-12) * forget_vec
            proj_forget_on_retain = (forget_vec @ retain_vec) / (retain_vec @ retain_vec + 1e-12) * retain_vec
            loss_r_grads = unflatten_grads(proj_retain_on_forget, retain_grads)
            loss_f_grads = unflatten_grads(proj_forget_on_retain, forget_grads)
            with torch.no_grad():
                for p, g_r, g_f in zip(trainable_params, loss_r_grads, loss_f_grads):
                    p.grad =  (g_r + g_f)
            optimizer_G.step()
        else:
            if use_gradient_ascent:
                loss = loss_retain + loss_anti_id* other_methods_lambda
            elif use_RMU:
                loss = loss_retain + loss_RMU* other_methods_lambda
            elif use_gmu:
                loss = loss_retain + loss_gmu_penalty* other_methods_lambda
            else:
                loss = loss_retain + loss_forget
            if mask_use:
                loss = loss + loss_mask 
            optimizer_G.zero_grad()
            if mask_use:
                mask_optimizer.zero_grad()
            loss.backward()
            optimizer_G.step()
            if mask_use:
                mask_optimizer.step()



        if i % 100 == 0:
            if eval_similar:
                snapshot_data = dict()
                snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
                snapshot_dir = os.path.join(ckpt_dir, "snapshots")
                os.makedirs(snapshot_dir, exist_ok=True)
                ckpt_path = os.path.join(snapshot_dir, f"step_{i:06d}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(snapshot_data, f)
            with torch.no_grad():
                generator.eval()
                if record_output:
                    intermediate_outputs = {}
                    register_hooks(generator.backbone.synthesis, intermediate_outputs)
                img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                if record_output:
                    torch.save(intermediate_outputs, os.path.join(output_dir, f"intermediate_outputs_{str(i).zfill(5)}.pt"))
                    remove_hooks()
                img_u_save = tensor_to_image(img_u_save)
                img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
                if mask_use:
                    mask_net.eval()
                    mask_u_save = mask_net(w_u)
                    img_u_save_mask = generator.synthesis(w_u*mask_u_save, camera_params_front)["image"]
                    img_u_save_mask = tensor_to_image(img_u_save_mask)
                    img_u_save_mask.save(os.path.join(image_dir, f"img_mask_{str(i).zfill(5)}.png"))
                img_retain = generator.synthesis(w_retain, camera_params_front)["image"]
                img_retain = tensor_to_image(img_retain)
                img_retain.save(os.path.join(image_dir, f"img_retain_{str(i).zfill(5)}.png"))
                generator.train()
                if mask_use:
                    mask_net.train()
                    del img_u_save_mask
            # del img_u_save, img_target_save
            del img_u_save

    with torch.no_grad():
        generator.eval()
        img_u_save = generator.synthesis(w_u, camera_params)["image"]
        img_target_save = g_source.synthesis(w_target, camera_params)["image"]
        img_u_save = tensor_to_image(img_u_save)
        img_target_save = tensor_to_image(img_target_save)
        img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
        img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
        generator.train()

    with torch.no_grad():
        generator.eval()
        if inversion is None: # for random
            for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                img_u = generator.synthesis(w_u, camera_params_view)["image"]
                img_u = tensor_to_image(img_u)
                img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
        else:
            if os.path.isdir(inversion_image_path): # for OOD
                for i in range(len(filenames)):
                    for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                        camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
            else: # for InD
                for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
        generator.train()

    snapshot_data = dict()
    snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
    with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
        pickle.dump(snapshot_data, f)
    if mask_use:
        torch.save(mask_net.state_dict(), os.path.join(ckpt_dir, f"mask_net.pth"))
    save_loss_record(loss_record, file_path=os.path.join(ckpt_dir, f"loss_record.csv"))
        
    
if __name__ == "__main__":
    unlearn() # pylint: disable=no-value-for-parameter