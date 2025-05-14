from transformers import pipeline
import gradio as gr

# Carrega o modelo de Q&A
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer(question, context):
    return qa(question=question, context=context)["answer"]

demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Pergunta"),
        gr.Textbox(label="Contexto", value="Este app responde perguntas sobre um texto.")
    ],
    outputs=gr.Textbox(label="Resposta"),
    title="AI Q&A Demo"
)

if __name__ == "__main__":
    demo.launch()
