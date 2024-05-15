import torch
from torchvision import transforms

from img_classifier import resnet18


def image2class(image_pil):

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = resnet18(num_classes=6)
    model.load_state_dict(torch.load("/home/suneo/catkin_ws/src/recognition_under_occ/weights/ResnetClassifier.pth", map_location=torch.device('cpu')))
    # prediction
    model.eval()
    with torch.no_grad():
        img = data_transform(image_pil)
        img_list = [img]
        batch_img = torch.stack(img_list, dim=0)

        # predict class
        output = model(batch_img).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

    del model

    return int(classes.numpy()[0])
