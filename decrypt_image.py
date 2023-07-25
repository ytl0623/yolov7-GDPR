import gradio as gr

def time(day,hour,minute):
    return "./output/{}/{}/{}/".format(day, hour, minute)

demo = gr.Interface(fn=time, 
                    inputs=[
                        gr.components.Textbox(label="time ex:2023-5-3"),
                        gr.components.Textbox(label="hour ex:1,2,3...,24"),
                        gr.components.Textbox(label="minute ex:1,2,3...,59"),
                    ],                
                    outputs="text")
    
if __name__ == "__main__":
    demo.launch()   



