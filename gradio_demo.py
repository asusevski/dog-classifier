 import gradio as gr

 def classify_image():
    # Load model
    


gr.Interface(fn=classify_image, 
    inputs=gr.Image(shape=(224, 224)),
    outputs=gr.Label(num_top_classes=3),
    examples=["banana.jpg", "car.jpg"]
).launch()