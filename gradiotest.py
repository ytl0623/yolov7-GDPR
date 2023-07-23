import cv2
import gradio as gr

def video_to_gray(input_video):
    # Read the input video
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = "output_video_gray.avi"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Write grayscale frame to output video
        out.write(gray_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_video_path

# Define the Gradio interface
inputs = gr.Video(type="file", label="Input Video")
outputs = gr.Video(type="file", label="Output Grayscale Video")
interface = gr.Interface(fn=video_to_gray, inputs=inputs, outputs=outputs)

# Launch the interface on a public web address
interface.launch()

