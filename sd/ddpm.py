import torch
import numpy as np

#use linear scheduler 
class DDPMSampler:

    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end = 0.0120):
        # 1000 numbers between beta_start and beta_end
        self.beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32).pow(2)
        self.alphas = 1.0 - self.beta 
        self.alpha_crumpod = torch.crumpod(self.alphas, dim=0) #[a0, a0*a1, a0*a1*a2, ...]

        self.one = torch.tensor(1.0)
        self.genrator = generator
        self.num_training_steps = num_training_steps
        #khi reverse, phai copy() de tao ra 1 array moi trc khi from_numpy
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    
    def set_inference_steps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps

        #999, 998, 997, ... 0
        #50: 999, 999-20, 999-40, ... 0
        step_ratio = self.num_training_steps // self.num_inference_steps
        time_steps = (np.arange(0, self.num_inference_steps) * step_ratio)[::-1].round().copy().astype(np.int64)
        self.timesteps = torch.from_numpy(time_steps)


    def add_noise(self, original_sample, timesteps):
        alpha_crumpod = self.alpha_crumpod.to(deivce=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alpha_crumpod[timesteps] ** 0.5 
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        std = (1 - alpha_crumpod[timesteps]) ** 0.5
        std = std.flatten()
        while len(std.shape) < len(original_sample.shape):
            std = std.unsqueeze(-1)

        mean = sqrt_alpha_prod * original_sample

        #according to equation (4) from the original paper 
        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, dtype=original_sample.dtype)
        noise_samples = mean + std * noise 
        return noise_samples 

    def _get_previous_timestep(self, t):
        prev_t = t - (self.num_training_steps // self.num_inference_steps)
        return prev_t 

    def _get_variance(self, timestep):
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_crumpod[timestep]
        alpha_prod_t_prev = self.alpha_crumpod[prev_t] if prev_t >=0 else self.one

        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        #compute using formula (7) in the paper 
        variance = current_beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        # ensure variance is not == 0
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[:start_step]
        self.start_step = start_step
    
    def step(self, timestep, latents, model_output):
        #model output : eps theta 
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_crumpod[timestep]
        alpha_prod_t_prev = self.alpha_crumpod[prev_t] if prev_t >= 0 else self.one 
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute the predicted original sample using formula (15) of the paper 
        pred_original_sample = (latents - (beta_prod_t**0.5) * model_output) / alpha_prod_t**0.5

        #compute the coeeficients for pred_original_sample and current sample x_t 
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t

        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        #compute the predicted previous sample mean
        pred_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device 
            noise = torch.randn(model_output.shape, generator=self.genrator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        #N(0m 1) -> n(mean, sigma**2)
        #x = mean + sigma * z where z ~ (0, 1)
        pred_previous_sample = pred_mean + variance
        return pred_previous_sample
        
            
        



