import os

class PromptConfig:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Model files, path relative to project root
        self.models = {
            "Model-1": os.path.join(BASE_DIR, "..", "model.pt"),
            "Model-2": os.path.join(BASE_DIR, "..", "best.pt")
        }

        # Example translation methods
        self.methods = {
            "Google": "google",
            "Helsinki-NLP's opus-mt-en-id model": "hf",
            "Gemini AI" : "gemini",
            "DeepL" : "deepl",
        }

        # Font files
        self.fonts = {
            "animeace_i": os.path.join(BASE_DIR, "..", "fonts", "fonts_animeace_i.ttf"),
            "mangati": os.path.join(BASE_DIR, "..", "fonts", "fonts_mangati.ttf"),
            "ariali": os.path.join(BASE_DIR, "..", "fonts", "fonts_ariali.ttf"),
        }
