import gradio as gr
from utils import configs

config = configs.PromptConfig()
with gr.Blocks(theme='JohnSmith9982/small_and_pretty', title= "Komik Translator",) as ui:
    gr.Markdown("Translate komik dari Inggris => Indonesia")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_link = gr.Textbox(label= "link mangadex", placeholder = "Masukan link disini!")
                button_link = gr.Button("Translate manga dengan link ini!", variant= "primary")
            with gr.Group():
                input_files = gr.Files(file_count= "multiple", file_types= ["image", ".zip", ".rar", ".pdf"])
                input_model = gr.Dropdown(
                                choices= list(config.models.keys()),
                                label="Model YOLO",
                                value="Model-2",
                                interactive=True,
                            )
                input_tl_method = gr.Dropdown(
                                    choices= list(config.methods.keys()),
                                    label="Translation Method",
                                    value="Google",
                                    interactive= True,
                                )
                input_font = gr.Dropdown(
                                choices= list(config.fonts.keys()),
                                label="Text Font",
                                value="animeace_i",
                                interactive=True,
                            )
                submit_button = gr.Button("Translate", variant= "primary")

        with gr.Column():
            ori_imgs= gr.Gallery(label="Gambar Asli"),
            result_imgs = gr.Gallery(label=" Hasil Terjemahan"),
            result_file = gr.File(label="Download File", visible= False)

    button_link.click (
        predict,
        inputs= [input_files,input_model,input_tl_method,input_font],
        outputs= [ori_imgs, result_imgs,result_file],
    )
    submit_button.click(
        predict,
        inputs= [input_files,input_model,input_tl_method,input_font],
        outputs= [ori_imgs, result_imgs,result_file],
    )
                
ui.launch(share=True)