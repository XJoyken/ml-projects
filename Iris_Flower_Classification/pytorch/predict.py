import torch
import cv2
from torchvision import transforms
from model import FlowersCNN


class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowersCNN().to(device=device)

model.load_state_dict(torch.load('D:\Code\ML\ml_projects\Iris_Flower_Classification\pytorch\\flowers_model_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return 'Image not found'
    
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img_bgr)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        max_prob, predicted_idx = torch.max(probabilities, 0)

        flower_name = class_names[predicted_idx.item()]
        confidence = max_prob.item() * 100

        print('Image: ', image_path)
        print(f'Prediction: {flower_name.upper()} (Confidence {confidence:.1f})%')
        print('-'*40)

predict_image('D:\\Code\ML\ml_projects\Iris_Flower_Classification\\flowers_predict\\rose\\rose1.jpg')
predict_image('D:\\Code\ML\ml_projects\Iris_Flower_Classification\\flowers_predict\\tulip\\tulip1.png')