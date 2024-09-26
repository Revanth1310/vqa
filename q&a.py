from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Get user input
img_path = "image.png"
question = "what is in the image?"

# Load and process the image
raw_image = Image.open(img_path).convert('RGB')
inputs = processor(raw_image, question, return_tensors="pt")

# Generate and decode the answer
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Answer:", answer)
