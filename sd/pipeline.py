import torch 
import numpy as np 
from tqdm import tqdm
from ddpm import DDPMSampler 

WIDTH = 512 
HEIGHT = 512 
LATENT_WIDTH = 512 //  8 
LATENT_HEIGHT = 512 // 8 

#strength: how much attention you want to pay to the initial image , the more, the more noise we add
#do_cfg: how much attention you want to pay to the condition prompt (1, 14)
def generate( prompt, 
             uncond_prompt,  #negative or empty prompt
             input_image=None, 
             strength=0.8, #how much noise -> more generative
             do_cfg=True, 
             cfg_scale=7.5, #how much attention you want to pay to the condition prompt
             sampler_name="ddpm", #scheduler
             n_inference_step=50, 
             models={}, 
             seed=None, 
             device=None, 
             idle_device=None, 
             tokenizer=None ):
    
#we are using the pretrained model
    with torch.no_grad():
        if not (0 < strength <= 1.0):
            raise ValueError("strength must be in the range (0, 1]")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # create a rabdom number generator
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            #convert prompt into token 
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            #batch_size, len
            cond_tokens = torch.tensor(cond_tokens, dytpe=torch.long, device=device)
            #batch_size, len, embed_dim
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            #batch_size, len, embed_dim
            uncond_tokens = clip(uncond_tokens)

            #(2, seq_len, embed_dim) = 2, 77, 768
            context = torch.cat([cond_context, uncond_tokens])

        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps (n_inference_step)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder = encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            #height, width, channel: 512, 512, 3
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            input_image_tensor = rescale(input_image_tensor, (0, 512), (-1,1))
            #height, width, channel -> batch, height, width, channel
            input_image_tensor = input_image_tensor.unsqueeze(0)
            #batch, height, width, channel -> batch, channel, height width
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            latents = encoder(input_image_tensor, encoder_noise)


            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

        else:
            #if we are doing test to image -> start with random noise N (0,1)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        #1000: 999, ...0
        #50: 1000, 980, 960, ... 0, each of the time step indicate the noise level

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #(1, 320)
            #current state of latents, prompt, time 
            time_embedding = get_time_embedding(timestep).to(device)

            #batch, 4, 64, 64
            model_input = latents

            if do_cfg:
                #batch,4 ,4, 64 -> batch*2, 4, 4, 64
                #1 use with prompt, one without prompt
                model_input = model_input.repeat(2, 1, 1, 1)

            #model output is the predicted noise by the UNET 
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove the noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        #batch, channel, height, width -> batch, height, width, channel
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=True):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min 
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    #[None]: unsqueeze 
    #(1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    #(1, 320)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)




