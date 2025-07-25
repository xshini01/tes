from utils.add_text import add_text
from utils.detect_bubbles import detect_bubbles
from utils.process_bubble import process_bubble
from utils.qwen2_vl_ocr import qwen2_vl_ocr
from utils.extract_file import extract_file
from utils.compress_toPDF import compress_toPDF
from utils import gemini_ai
from utils.mangadex_downloader import mangadex_download
from utils.translator import MangaTranslator
from utils import configs
from IPython.display import clear_output
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import gradio as gr
import subprocess
from google.genai.errors import ClientError
import cv2
import time
import os
from tqdm import tqdm
import shutil
from natsort import natsorted

config = configs.PromptConfig()

def split_ocr_by_semicolon(ocr_text):
    lines = [line.strip() for line in ocr_text.strip().split('\n') if line.strip()]
    segments = []
    current = []

    for line in lines:
        if line.endswith(";"):
            current.append(line.rstrip(";"))
            segments.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)

    if current:
        segments.append("\n".join(current).strip())

    return segments


def combine_bubbles_vertically(cropped_images):
    widths, heights = zip(*(img.size for img in cropped_images))
    max_width = max(widths)
    total_height = sum(heights) + 10 * (len(cropped_images) - 1)

    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in cropped_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.size[1] + 10

    return combined_image

def get_images(image_folder):
    image_paths = [
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return image_paths  # Mengembalikan daftar path gambar

# fungsi mencoba api
def retry_on_429(func, *args, max_retries=10, base_wait=5, **kwargs):
    """Retry jika terjadi error 429 (RESOURCE_EXHAUSTED) atau 503 (UNAVAILABLE) dengan exponential backoff."""
    retries = 0

    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            error_message = str(e)
            error_code = getattr(e, 'code', None)

            if 'RESOURCE_EXHAUSTED' in error_message or '429' in error_message or error_code == 429:
                retries += 1
                wait_time = base_wait * (2 ** (retries - 1)) 
                print(f"Token habis. Coba lagi dalam {wait_time} detik... ({retries}/{max_retries})")
                time.sleep(wait_time)
            elif 'UNAVAILABLE' in error_message or '503' in error_message or error_code == 503:
                retries += 1
                wait_time = base_wait * (2 ** (retries - 1)) 
                print(f"Model unavailable. Coba lagi dalam {wait_time} detik... ({retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise  
        except Exception as e:
            print(f"Error lain: {e}")
            break

    raise RuntimeError(f"Gagal setelah {max_retries} percobaan.")


# main fungsi
def predict(files_input, model, translation_method, font, progress=gr.Progress(track_tqdm=True)):
    source_dir = 'folder_ekstrak'
    save_dir = "save_images"

    MODEL = config.models.get(model, "best.pt")
    font = config.fonts.get(font, "fonts/fonts_animeace_i.ttf")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(files_input, str) and files_input.startswith(("http://", "https://")):
        mangadex_download(files_input)
    else:
        extract_file(files_input)

    manga_translator = MangaTranslator()

    for root, dirs, files in os.walk(source_dir):
        files = natsorted(files)
        for file in tqdm(files, desc="Memproses Gambar"):
            file_path = os.path.join(root, file)
            results = detect_bubbles(MODEL, file_path)
            image = cv2.imread(file_path)

            bubbles_data = []

            for result in results:
                x1, y1, x2, y2, score, class_id = result
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # 1. crop asli → untuk OCR
                original_crop = image[y1:y2, x1:x2]
                pil_crop = Image.fromarray(original_crop)

                # 2. proses bubble (hapus teks) → untuk pasang teks baru
                processed_crop, bubble_cont = process_bubble(original_crop)

                bubbles_data.append({
                    'coords': (x1, y1, x2, y2),
                    'original_crop': pil_crop,
                    'processed_crop': processed_crop,
                    'bubble_cont': bubble_cont
                })

            if not bubbles_data:
                continue

            # === OCR batch: gabungkan original crop (belum dibersihkan) ===
            combined_image = combine_bubbles_vertically([b['original_crop'] for b in bubbles_data])
            combined_path = "combined_bubbles.png"
            combined_image.save(combined_path)

            if gemini_ai.genai_token:
                ocr_result = retry_on_429(gemini_ai.gemini_ai_ocr, combined_path)
            else:
                ocr_result = qwen2_vl_ocr(combined_image, model_ocr, processor_ocr)
            
            print(f"ini hasil ocr {ocr_result}")
            # === Ubah jadi pisah per speech bubble berdasarkan tanda ;
            ocr_lines = split_ocr_by_semicolon(ocr_result)
            print(f"ini ocr line {ocr_lines}")

            # Tambahkan dummy jika hasil kurang dari jumlah bubble
            while len(ocr_lines) < len(bubbles_data):
                ocr_lines.append("[OCR MISS]")

            # === Translasi semua hasil OCR ===
            translated_lines = []
            for line in ocr_lines:
                if gemini_ai.genai_token:
                    translated = retry_on_429(gemini_ai.gemini_ai_translator, line)
                else:
                    translated = manga_translator.translate(line, method=translation_method)
                translated_lines.append(translated)

            # === Tempel hasil ke gambar asli ===
            for bubble, translated in zip(bubbles_data, translated_lines):
                x1, y1, x2, y2 = bubble['coords']
                final_crop = add_text(bubble['processed_crop'], translated, font, bubble['bubble_cont'])
                image[y1:y2, x1:x2] = final_crop

            output_idx = sum(1 for entry in os.scandir(save_dir) if entry.is_file()) + 1
            output_path = os.path.join(save_dir, f"output_image_{output_idx}.png")
            cv2.imwrite(output_path, image)
            time.sleep(0.1)


    to_pdf= compress_toPDF()

    return get_images(source_dir), get_images(save_dir), gr.update(value=to_pdf, visible=True)

# ui token
with gr.Blocks() as token_interface:
        gr.Markdown("## Token/API Key Gemini Ai (opsional)")
        token_input = gr.Textbox(
            label="Jika Anda menggunakan Token/API Key Gemini AI, OCR dan terjemahan akan dilakukan menggunakan Gemini AI. Jika tidak, model default(qwen2_vl_ocr) akan digunakan.",
            info="Anda bisa mendapatkan Token/API Key Gemini AI dari <a href=\"https://aistudio.google.com/apikey\" target=\"_blank\"> SINI (Token/API Key Google)</a>. \nToken/API Key ini bersifat opsional dan dapat digunakan untuk pemindaian dan terjemahan teks menggunakan Gemini AI (Google).",
            placeholder="Masukan Token/API Key disini (opsional) ...",
            type="password"
        )
        save_button = gr.Button("Submit", variant="primary")
        output_label = gr.Label(label= "your Token/API Key :")
        save_button.click(fn=gemini_ai.save_token, inputs=token_input, outputs=output_label)

clear_output()
token_interface.launch(share=True)

while not gemini_ai.token_set:
    time.sleep(2)

model_ocr, processor_ocr = None, None

if not gemini_ai.genai_token:
    def load_ocr_model():
        global model_ocr, processor_ocr
        if model_ocr is None or processor_ocr is None:
            model_ocr = Qwen2VLForConditionalGeneration.from_pretrained(
                "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
            )
            processor_ocr = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
    
    load_ocr_model()

# main interface
with gr.Blocks(theme='JohnSmith9982/small_and_pretty', title="Komik Translator") as ui:
    gr.Markdown("Translate komik dari Inggris => Indonesia")

    with gr.Row():
        with gr.Column(variant='panel') as content_group:
            input_mode = gr.Radio(
                ["Input file/gambar", "Input link MangaDex"],
                value="Input file/gambar",
                label="Pilih metode input",
                info="Pilih salah satu metode input: upload files/gambar atau link MangaDex",
                interactive=True
            )

            with gr.Group(visible=True) as content_file:
                input_files = gr.Files(file_count="multiple", file_types=["image", ".zip", ".rar", ".pdf"])
                input_model = gr.Dropdown(
                    choices=list(config.models.keys()),
                    label="Model YOLO",
                    value="Model-2",
                    interactive=True,
                )
                input_tl_method = gr.Dropdown(
                    choices=list(config.methods.keys()),
                    label="Translation Method",
                    value="Google",
                    interactive=True,
                )
                input_font = gr.Dropdown(
                    choices=list(config.fonts.keys()),
                    label="Text Font",
                    value="animeace_i",
                    interactive=True,
                )
                submit_button = gr.Button("Translate", variant="primary")

            with gr.Column(variant='panel', visible=False) as content_link:
                input_link = gr.Textbox(label="Link MangaDex", placeholder="Masukan link disini!")
                button_link = gr.Button("Translate manga dengan link ini!", variant="primary")

        def show_mode(mode):
            if mode == "Input link MangaDex":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        input_mode.change(show_mode, inputs=input_mode, outputs=[content_link, content_file])

        with gr.Column(variant='panel'):
            ori_imgs = gr.Gallery(label="Gambar Asli")
            result_imgs = gr.Gallery(label="Hasil Terjemahan")
            result_file = gr.File(label="Download File", visible=False)

    button_link.click(
        predict,
        inputs=[input_link, input_model, input_tl_method, input_font],
        outputs=[ori_imgs, result_imgs, result_file],
    )
    submit_button.click(
        predict,
        inputs=[input_files, input_model, input_tl_method, input_font],
        outputs=[ori_imgs, result_imgs, result_file],
    )

clear_output()
ui.launch(debug=True, share=True, inline=False)
