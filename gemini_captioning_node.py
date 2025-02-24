import time
import os
import io
import base64
from PIL import Image
import numpy as np
import google.generativeai as genai

class GeminiCaptioningNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "PROMPT TYPE": (("SD1.5 – SDXL", "FLUX"), {"default": "SD1.5 – SDXL"}),
                "APY KEY PATH": ("STRING", {"default": ""}),
            },
            "optional": {
                "PROMPT LENGTH": ("INT", {"default": 0, "defaultInput": True}),
                "PROMPT REFERENCE": ("STRING", {"default": "", "defaultInput": True}),
                "PROMPT STRUCTURE": ("STRING", {"default": "", "defaultInput": True}),
                "IGNORE": ("STRING", {"default": "", "defaultInput": True}),
                "EMPHASIS": ("STRING", {"default": "", "defaultInput": True}),
                "SAVE TO PATH": ("STRING", {"default": "", "defaultInput": True}),
                "TXT NAME": ("STRING", {"default": "", "defaultInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESULT", "REQUEST TEXT", "LOG")
    FUNCTION = "gemini_caption"
    CATEGORY = "AS_GeminiCaptioning"

    def gemini_caption(self, **kwargs):
        log = []
        # Retrieve parameters from kwargs and default to empty strings if None.
        image = kwargs.get("IMAGE")
        prompt_type = kwargs.get("PROMPT TYPE") or ""
        apy_key_path = kwargs.get("APY KEY PATH") or ""
        prompt_length = kwargs.get("PROMPT LENGTH")
        prompt_reference = kwargs.get("PROMPT REFERENCE") or ""
        prompt_structure = kwargs.get("PROMPT STRUCTURE") or ""
        ignore = kwargs.get("IGNORE") or ""
        emphasis = kwargs.get("EMPHASIS") or ""
        save_to_path = kwargs.get("SAVE TO PATH") or ""
        txt_name = kwargs.get("TXT NAME") or ""
        
        # Process the input image and convert to a PIL Image, then obtain its data.
        try:
            if isinstance(image, bytes):
                image_data = image
                img_obj = Image.open(io.BytesIO(image))
            elif hasattr(image, "read"):
                image_data = image.read()
                img_obj = Image.open(io.BytesIO(image_data))
            elif hasattr(image, "cpu") and hasattr(image, "detach"):
                # Assume image is a PyTorch tensor.
                tensor = image.cpu().detach().numpy()
                # Remove extra leading singleton dimensions.
                while tensor.ndim > 3 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                # Convert from CHW to HWC if applicable.
                if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
                    tensor = tensor.transpose(1, 2, 0)
                # Scale values if needed.
                if tensor.max() <= 1:
                    tensor = (tensor * 255).astype("uint8")
                else:
                    tensor = tensor.astype("uint8")
                img_obj = Image.fromarray(tensor)
                buffer = io.BytesIO()
                fmt = "PNG"
                img_obj.save(buffer, format=fmt)
                image_data = buffer.getvalue()
            elif isinstance(image, np.ndarray):
                # Remove extra leading singleton dimensions.
                while image.ndim > 3 and image.shape[0] == 1:
                    image = image[0]
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = image.transpose(1, 2, 0)
                if image.max() <= 1:
                    image = (image * 255).astype("uint8")
                else:
                    image = image.astype("uint8")
                img_obj = Image.fromarray(image)
                buffer = io.BytesIO()
                fmt = "PNG"
                img_obj.save(buffer, format=fmt)
                image_data = buffer.getvalue()
            else:
                # Assume image is a PIL Image.
                img_obj = image
                buffer = io.BytesIO()
                fmt = image.format if image.format else "PNG"
                image.save(buffer, format=fmt)
                image_data = buffer.getvalue()
            detected_format = img_obj.format if img_obj.format else "PNG"
            log.append(f"Detected image format: {detected_format}")
        except Exception as e:
            log.append("Error processing image: " + str(e))
            return ("", "", "\n".join(log))
        
        # Define built-in prompt templates.
        SDXL_type = ("It should be a text in CLIP-L encoder format (concise, comma-separated keywords, around 20-40 words). Here’s an example: «Architecture, high-end modernist residential complex, minimalist design, open balconies, subtle architectural details, concrete and glass façades, elegant geometric volumes, tiered rooftop terraces, panoramic floor-to-ceiling windows. Neutral-toned stone panels, tinted glass curtain walls, brushed metal railings, integrated with lush landscaping, manicured hedges, ornamental grasses, sculptural trees, wooden pathway leading to a reflective metal sphere. Secluded urban oasis, tranquil environment, free from city noise, surrounded by curated greenery, creating a serene and balanced atmosphere.Soft diffused lighting, overcast sky, early morning mist, gentle atmospheric glow, cinematic wide-angle perspective, symmetrical framing, high dynamic range, RAW photo, hyper-detailed, photorealistic»")
        FLUX_type = ("It should be a text in T5 encoder format (detailed, natural language, around 100 words). Here’s an example: «Architecture, high-end modernist residential complex surrounded by lush greenery, designed with a minimalist and elegant aesthetic. The buildings feature a combination of natural stone and glass façades, with subtle architectural details and open balconies. A linear yet dynamic composition with clean geometric volumes, softened by carefully curated landscaping, including hedges, ornamental grasses, and small trees. The façade combines smooth concrete panels with floor-to-ceiling tinted glass windows, creating a refined balance of opacity and transparency. The outdoor space is defined by a wooden pathway meandering through a meticulously designed garden, leading towards a focal point—a polished metal sphere sculpture. Strategic lighting elements subtly highlight the landscape, while the gentle play of reflections on the glass surfaces enhances the depth of the environment. Set in a tranquil urban enclave, free from visual noise, framed by an overcast sky that casts a soft, diffused glow over the buildings. Early morning atmosphere with slight fog in the distance, lending an ethereal and cinematic quality to the scene. RAW photo, slightly elevated wide-angle viewpoint, long exposure, cinematic framing, balanced symmetry, moderate depth of field, high dynamic range, hyper-detailed, photorealistic rendering»")
        
        # Assemble the text prompt.
        blocks = []
        blocks.append(
            "Give me a description of this image in English in the format of a text prompt for Stable Diffusion. It should be only the descriptive text according to the template I provided, without any additional comments from you. The text should be continuous, without headings, lists, or any other formatting."
        )
        ref_text = prompt_reference.strip() if prompt_reference.strip() != "" else (SDXL_type if prompt_type == "SD1.5 – SDXL" else FLUX_type)
        blocks.append("Use the following reference as an example of the prompt format and structure, showing how the text should look. Use it only as a reference, do not use its content for the current request unless it is present in the attached image: \n" + ref_text)
        default_structure = "1) Type of building, 2) Shape of the building, 3) Materials, 4) Location and surroundings, 5) Season, weather, time of day, lighting, 6) Camera position and angle, composition, camera parameters"
        structure_text = prompt_structure.strip() if prompt_structure.strip() != "" else default_structure
        blocks.append("The structure of the prompt should be as follows (do not create headings or comments, only follow the order of information in the description):\n" + structure_text)
        if ignore.strip() != "":
            blocks.append("In the prompt, be sure to ignore any mention of anything related to: " + ignore.strip())
        if emphasis.strip() != "":
            blocks.append("In the prompt, emphasize additional attention on: " + emphasis.strip())
        if isinstance(prompt_length, int) and prompt_length > 0:
            blocks.append("The approximate length of the prompt should be at least " + str(prompt_length) + " words")
        
        prompt_text = "\n".join(blocks)
        log.append("Constructed prompt text successfully.")
        
        # Read and configure the API key.
        try:
            with open(apy_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            log.append("API key read successfully from file.")
        except Exception as e:
            log.append("Error reading API key file: " + str(e))
            return ("", prompt_text, "\n".join(log))
        
        genai.configure(api_key=api_key)
        
        # Encode image to base64.
        try:
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            log.append("Image encoded to base64 successfully.")
        except Exception as e:
            log.append("Error encoding image: " + str(e))
            return ("", prompt_text, "\n".join(log))
        
        # Determine MIME type based on detected image format.
        fmt_upper = detected_format.upper()
        if fmt_upper == "PNG":
            mime_type = "image/png"
        elif fmt_upper in ("JPG", "JPEG"):
            mime_type = "image/jpeg"
        elif fmt_upper == "WEBP":
            mime_type = "image/webp"
        elif fmt_upper in ("HEIC", "HEIF"):
            mime_type = "image/heic"
        else:
            mime_type = "image/png"
        log.append(f"Using MIME type: {mime_type}")
        
        # Build payload as a list: first element is the image dict, then the prompt text.
        payload = [
            {
                "mime_type": mime_type,
                "data": encoded_image,
            },
            prompt_text,
        ]
        log.append("Payload prepared for Gemini API request.")
        
        # Call the Gemini API using the google-generativeai SDK.
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            log.append("Initiating API call to Gemini...")
            response = model.generate_content(payload, request_options={"timeout": 600})
            result_text = response.text if response.text is not None else ""
            log.append("Gemini API call completed successfully.")
        except Exception as e:
            log.append("Error during Gemini API call: " + str(e))
            result_text = "Error: " + str(e)
        
        # Optionally save the result to file.
        if save_to_path.strip() != "":
            try:
                filename = txt_name.strip() if txt_name.strip() != "" else "result.txt"
                # Ensure the filename ends with .txt
                if not filename.lower().endswith(".txt"):
                    filename += ".txt"
                full_path = os.path.join(save_to_path.strip(), filename)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(result_text)
                log.append("Result saved to file: " + full_path)
            except Exception as e:
                log.append("Error saving result to file: " + str(e))
        
        return (result_text, prompt_text, "\n".join(log))
