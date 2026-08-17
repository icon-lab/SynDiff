
import argparse
import torch
import numpy as np, h5py

import os
import torch.optim as optim
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp
from diffusion import (Posterior_Coefficients, get_time_schedule,
                       sample_from_model)
from dataset import CreateDatasetSynthesis

import torch.nn.functional as F

import torchvision.transforms as transforms

def psnr(img1, img2):
    #Peak Signal to Noise Ratio

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))


def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    # Checkpoints carry one 'module.' prefix per parallel wrapper the saving
    # run used: none for a single-process run, one for DistributedDataParallel.
    # Blindly dropping the first 7 characters corrupted every key of an
    # unprefixed checkpoint.
    prefix = 'module.'
    normalised = {}
    for key, value in ckpt.items():
        while key.startswith(prefix):
            key = key[len(prefix):]
        normalised[key] = value
    netG.load_state_dict(normalised)
    netG.eval()
#%%
def sample_and_test(args):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_chose)
        device = torch.device('cuda:{}'.format(args.gpu_chose))
    else:
        device = torch.device('cpu')
    epoch_chosen=args.which_epoch
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    #loading dataset
    phase='test'
    dataset=CreateDatasetSynthesis('test', args.input_path, args.contrast1, args.contrast2, image_size=args.image_size)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4)
    #Initializing and loading network
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir,exp)

    checkpoint_file = exp_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, gen_diffusive_1,'gen_diffusive_1',epoch=str(epoch_chosen), device = device)
    load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch_chosen), device = device)


    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
         
    save_dir = exp_path + "/generated_samples/epoch_{}".format(epoch_chosen)
    
    # CreateDatasetSynthesis pads every volume out to 256x256; this crop undoes
    # that padding. The defaults match the IXI/BRATS geometry the paper used --
    # set --crop_h/--crop_w to your own slice size for other datasets.
    crop = transforms.CenterCrop((args.crop_h, args.crop_w))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    loss1 = np.zeros((1,len(data_loader)))
    loss2 = np.zeros((1,len(data_loader)))
    # collected per slice and stacked afterwards, so the stored volume always
    # matches the cropped image size
    syn_im1=[]
    syn_im2=[]
    for iteration, (x , y) in enumerate(data_loader): 
        
        real_data = x.to(device, non_blocking=True)
        source_data = y.to(device, non_blocking=True)
        
        x1_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
        #diffusion steps
        fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
    
        fake_sample1 = to_range_0_1(fake_sample1) ; fake_sample1 = fake_sample1/fake_sample1.max()
        real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.max()
        source_data = to_range_0_1(source_data); source_data = source_data/source_data.max() 
        
        
        fake_sample1 = crop(fake_sample1) 
        real_data = crop(real_data)
        source_data = crop(source_data) 
        syn_im1.append(np.squeeze(fake_sample1.cpu().numpy()))
        
        loss1[0, iteration] = psnr(fake_sample1, real_data).cpu().numpy()
        print(str(iteration))
        fake_sample1 = torch.cat((source_data, fake_sample1, real_data),axis=-1)
        torchvision.utils.save_image(fake_sample1, '{}/{}_samples1_{}.jpg'.format(save_dir, phase, iteration), normalize=True)

    for iteration, (x , y) in enumerate(data_loader): 
        
        real_data = y.to(device, non_blocking=True)
        source_data = x.to(device, non_blocking=True)
        
        x2_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
        #diffusion steps
        fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
    
        
        fake_sample2 = to_range_0_1(fake_sample2) ; fake_sample2 = fake_sample2/fake_sample2.max()
        real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.max()
        source_data = to_range_0_1(source_data); source_data = source_data/source_data.max() 
        
        
        
        fake_sample2 = crop(fake_sample2) 
        real_data = crop(real_data)
        source_data = crop(source_data)
        syn_im2.append(np.squeeze(fake_sample2.cpu().numpy())) 
        
        loss2[0, iteration] = psnr(fake_sample2, real_data).cpu().numpy()
        print(str(iteration))
        fake_sample2 = torch.cat((source_data, fake_sample2, real_data),axis=-1)
        torchvision.utils.save_image(fake_sample2, '{}/{}_samples2_{}.jpg'.format(save_dir, phase, iteration), normalize=True)

    print(np.nanmean(loss1))
    np.save('{}/psnr_values1.npy'.format(save_dir), loss1)

    print(np.nanmean(loss2))
    np.save('{}/psnr_values2.npy'.format(save_dir), loss2)

    syn_im1 = np.stack(syn_im1, axis=-1)
    syn_im2 = np.stack(syn_im2, axis=-1)

    f = h5py.File(save_dir + '/im_syn.mat',  "w")
    f.create_dataset('images_'+args.contrast1+'syn', data=syn_im1)
    f.create_dataset('images_'+args.contrast2+'syn', data=syn_im2)
    f.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--crop_h', type=int, default=256,
                        help='height the padded output is cropped back to')
    parser.add_argument('--crop_w', type=int, default=152,
                        help='width the padded output is cropped back to')
    parser.add_argument('--which_epoch', type=int, default=50)
    parser.add_argument('--gpu_chose', type=int, default=0)


    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')   
    args = parser.parse_args()
    
    sample_and_test(args)
    
