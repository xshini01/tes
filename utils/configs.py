class PromptConfig:
    def __init__(self):
        self.models = {
            "Model-1": "model.pt",
            "Model-2" : "best.pt"
        }
        self.methods = {
            "Google": "google",
            "DeepL": "deepl",
            "Helsinki-NLP's opus-mt-en-id model":"hf",
            "Baidu": "baidu",
            "Bing": "bing",
        }
        self.fonts = {
            "animeace_i": "fonts/fonts_animeace_i.ttf",
            "mangati": "fonts/fonts_mangati.ttf",
            "ariali": "fonts/fonts_ariali.ttf",
        }