from .gemini_captioning_node import GeminiCaptioningNode

NODE_CLASS_MAPPINGS = {
    "AS_GeminiCaptioning": GeminiCaptioningNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AS_GeminiCaptioning": "AS_GeminiCaptioning"
}

def load_plugin():
    try:
        from modules import nodes  # Adjust this import if your ComfyUI version differs
    except ImportError:
        print("Module 'modules.nodes' not found. Make sure you have ComfyUI installed.")
        return
    nodes.register_node(GeminiCaptioningNode)
    print("AS_GeminiCaptioning plugin successfully registered.")
