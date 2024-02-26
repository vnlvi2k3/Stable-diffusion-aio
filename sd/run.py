import model_loader
import pipeline
import torch
from PIL import Image 
from transformers import CLIPTokenizer 

DEVICE = "cpu"

ALLOW_CUDE = True 

if torch.cude.is_available() and ALLOW_CUDE:
    DEVICE = "cuda"

print(f"Using device {DEVICE}")

def main():
    tokenizer = CPILTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


    # text to image

    prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 8k resolution"
    uncond_prompt = ""
    do_cfg = True 
    cfg_scale = 7 

    #image to image 
    input_image = None 
    image_path = "../data/dog.jpg"
    # input_image = Image.open(image_path)
    strength = 0.9 

    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42 
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_step=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device=DEVICE,
        tokenizer=tokenizer
    )

    image_to_save = Image.fromarray(output_image)
    image_to_save.save("../data/output.png")


if __name__ == "__main__":
    main()