import json
import os
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_protect
import numpy as np
import torch
from django.conf import settings
from .models import StoredImage,StoredInternalImage
from .crawler import Fill_database
import requests
import requests
from PIL import Image
from io import BytesIO
from .clip_3 import CLIP,encode_img,tokenize_text,img_text_find,get_trained_clip,get_images,get_local_data
import torchvision.transforms as transforms
from PIL import Image as im 
import albumentations

def fetch_image(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Convert image content to PIL Image
            image = Image.open(BytesIO(response.content))

            image = image.convert('RGB')
            # Define transformations
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
            # Apply transformations
            input_tensor = preprocess(image)
            # Add batch dimension
            input_batch = input_tensor.unsqueeze(0)
            # Return tensor
            
            return input_batch
        else:
            print("Failed to fetch image. HTTP status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)
    



@csrf_protect
# Create your views here.


#def Update_local_dataset(request):


def Expand_dataset(request):
    if request.method == 'POST':
        prompt = str(json.loads(request.body)["prompt"])
        max_res = str(json.loads(request.body)["quantity"])
        search_results = Fill_database(prompt,max_res)
        my_model = get_trained_clip()
        for search_result in search_results:
            address = search_result[1]
            if not StoredImage.objects.filter(address=address).exists():
                image = fetch_image(search_result[1])
                image_enc = encode_img(image,my_model).cpu().tolist()
                print(f"here {len(image_enc[0])} image_enc {image_enc}")
                stored_image = StoredImage()
                stored_image.image_data = image_enc
                stored_image.caption = search_result[0]
                stored_image.address = address
                stored_image.save()
            print("Done")
        result = {'result 2': search_results}  # Replace this with your actual response data
        return JsonResponse(result)

def home(request):
    """
    stored_images = StoredImage.objects.all()
    for m in stored_images:
        m.delete()
    stored_images = StoredInternalImage.objects.all()
    for m in stored_images:
        m.delete()
    """
    return render(request, "base.html")

def login(request):
    stored_images = StoredImage.objects.all()
    for m in stored_images:
        m.delete()
    stored_images = StoredInternalImage.objects.all()
    for m in stored_images:
        m.delete()

    return render(request, "base.html")

def update_database(request):
    model = get_trained_clip().eval()
    dir_path = os.path.join(os.getcwd(), "my_clip" , "static" , "roco-dataset", "data", "train", "radiology", "images")
    local_data = get_local_data(model)
    for index, row in local_data.iterrows():
        print(f"index {index}")
        
        image_index = row['Image Index']
        address = f"{dir_path}/{image_index}"
        caption = row['captions']
        resize = albumentations.Resize(224, 224, always_apply=True)
        
        if not StoredInternalImage.objects.filter(address=address).exists():
            image = im.open(address) if dir else None
            image = torch.tensor(resize(image = np.array(image))["image"]).permute(2, 0, 1).float()
            image = image.unsqueeze(0)
            image_enc = encode_img(image,model).cpu().tolist()
            stored_image = StoredInternalImage()
            stored_image.image_data = image_enc
            stored_image.caption = caption
            stored_image.address = address
            stored_image.save()

def ret_image(request):
    if request.method == 'POST':
        dir_path = os.path.join(os.getcwd(), "my_clip" , "static" , "roco-dataset", "data", "train", "radiology", "images")
        my_response = {}
        model = get_trained_clip().eval()
        max_res = int(json.loads(request.body)["quantity"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prompt = str(json.loads(request.body)["prompt"])
        external = bool(json.loads(request.body)["external"])
        if external:
            stored_images = StoredImage.objects.all()
            my_tensors = []
            my_captions = []
            my_filenames = []
            for stored_image in stored_images:
                image_features = torch.tensor(stored_image.image_data)
                my_captions.append(stored_image.caption)
                my_filenames.append(stored_image.address)
                my_tensors.append(image_features)
         
            stacked_tensors = torch.stack(my_tensors, dim=0).squeeze().to(device)
            print(f"stacked_tensors {stacked_tensors.shape}")
            response = get_images(model,prompt,max_res,stacked_tensors,my_filenames,my_captions)
            print(len(response))
            for i,match in enumerate(response):
                my_response[i] = {'address': match[0], 'caption': match[1]}
        else:
            """
            stored_images = StoredInternalImage.objects.all()
            my_tensors = []
            my_captions = []
            my_filenames = []
            for stored_image in stored_images:
                image_features = torch.tensor(stored_image.image_data)
                my_captions.append(stored_image.caption)
                my_filenames.append(stored_image.address)
                my_tensors.append(image_features)
         
            stacked_tensors = torch.stack(my_tensors, dim=0).squeeze().to(device)
            print(f"stacked_tensors {stacked_tensors.shape}")
            response = get_images(model,prompt,max_res,stacked_tensors,my_filenames,my_captions)
            print(len(response))
            for i,match in enumerate(response):
                my_response[i] = {'address': settings.IMAGE_URL + match[0], 'caption': match[1]}

            """
            response = get_images(model,prompt,max_res)
            for i,match in enumerate(response):
                my_response[i] = {'address': settings.IMAGE_URL + match[0], 'caption': match[1]}
            
        return JsonResponse(my_response)
    else:
        # Handle GET requests or other methods if needed
        return JsonResponse({'error': 'Method not allowed'}, status=405)



