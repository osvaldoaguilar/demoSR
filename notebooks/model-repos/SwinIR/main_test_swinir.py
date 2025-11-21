import argparse
import cv2
import glob
import numpy as np
import os
import torch
import requests

from models.network_swinir import SwinIR as net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', 
                        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--save', type=str, default=None, help='absolute path to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    # Si se pasó --save, sobrescribe save_dir
    if args.save:
        save_dir = args.save
    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        imgname, img_lq, img_gt = get_image_pair(args, path)
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
        print('Processed {:d}: {:s}'.format(idx, imgname))


def define_model(args):
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    elif args.task == 'real_sr':
        if not args.large_model:
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6,6,6,6,6,6,6,6,6], embed_dim=240,
                        num_heads=[8,8,8,8,8,8,8,8,8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def setup(args):
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt if args.folder_gt else args.folder_lq
        border = args.scale
        window_size = 8
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model: save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_lq
        border = 0
        window_size = 8
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_lq
        border = 0
        window_size = 7
    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    imgname, imgext = os.path.splitext(os.path.basename(path))
    if args.task in ['classical_sr', 'lightweight_sr']:
        if args.folder_gt:
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(np.float32)/255.
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)/255.
        else:
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_lq is None:
                raise FileNotFoundError(f"No se pudo leer la imagen LQ: {path}")
            img_lq = img_lq.astype(np.float32)/255.
            img_gt = None
    elif args.task in ['real_sr']:
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)/255.
        img_gt = None
    elif args.task in ['gray_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise/255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)/255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise/255., img_gt.shape)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt.ndim != 2:
            from utils import util_calculate_psnr_ssim as util
            img_gt = util.bgr2ycbcr(img_gt, y_only=True)
        _, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32)/255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32)/255.
    elif args.task in ['color_jpeg_car']:
        img_gt = cv2.imread(path)
        _, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 1)
        img_gt = img_gt.astype(np.float32)/255.
        img_lq = img_lq.astype(np.float32)/255.
    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be multiple of window_size"
        stride = tile - args.tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*args.scale, w*args.scale).type_as(img_lq)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(out_patch)
                W[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(out_patch_mask)
        output = E.div_(W)
    return output


if __name__ == '__main__':
    main()
