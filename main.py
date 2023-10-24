import streamlit as st
import torchxrayvision as xrv
import cv2
import torch
import torchvision
import PIL
import numpy as np

def load_xray(data_path):
    image_loaded = PIL.Image.open(data_path)
    image_loaded = image_loaded.resize((350,350))
    img = np.array(image_loaded)
    show_XRAY(img)
    if img.ndim==2:
        img = np.expand_dims(img,axis=0)
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img.mean(2)[None, ...] # Make single color channel
        return img

    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel
    return  img

def findings(results,thresh):
    if  [(key, value) for key, value in results.items() if value > thresh]==[]:
        st.write('This XRAY is normal')
    else:
        findings_list= [(key, value) for key, value in results.items() if value > thresh]
        for finding in findings_list:
            st.write(finding[0],'with probability','{:.3f}'.format(finding[1])) 
        
#         st.write ([(key, value) for key, value in results.items() if value > thresh])
    
def show_XRAY(image):
    image = np.squeeze(image)
    st.image(image)
    
code = st.text_input("Enter code")
if code=='ekpa':
    # st.title("Triage of Chest X-RAYs")
    with st.container():
        bio_image= cv2.imread('ekpa.PNG')
        bio_image = cv2.cvtColor(bio_image, cv2.COLOR_BGR2RGB)
        st.image(bio_image)
    uploaded_file = st.file_uploader("Choose an XRAY image (not DICOM) ",type=['png', 'jpg','jpeg'])
    if uploaded_file is not None:
        img = load_xray(uploaded_file)
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img = torch.from_numpy(img)
        # Load model and process image
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        outputs = model(img[None,...]) # or model.features(img[None,...]) 
        # Print results
        results = dict(zip(model.pathologies,outputs[0].detach().numpy()))
        findings(results,0.6)
        
if code=='bio':
    st.title("Triage of Chest X-RAYs")
    with st.container():
        bio_image= cv2.imread('Bioiatriki.png')
        bio_image = cv2.cvtColor(bio_image, cv2.COLOR_BGR2RGB)
        st.image(bio_image)
    uploaded_file = st.file_uploader("Choose an XRAY image (not DICOM) ",type=['png', 'jpg','jpeg'])
    if uploaded_file is not None:
        img = load_xray(uploaded_file)
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img = torch.from_numpy(img)
        # Load model and process image
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        outputs = model(img[None,...]) # or model.features(img[None,...]) 
        # Print results
        results = dict(zip(model.pathologies,outputs[0].detach().numpy()))
        findings(results,0.6)
