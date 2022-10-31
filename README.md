# UDA_thermal

In this work, we propose an Adversarial UDA model to solve the Hand gesture recognition task. UDA is used so that the model can leverage the information from the source domain (RGB) and improve the model's performance on the Target domain (thermal).

# Datasets
1. Sign digit dataset - https://drive.google.com/drive/folders/1maCq5fEn9kwOYBMc9OBJVstcAygUBsDv?usp=share_link

2. Alphabet gesture classification dataset: \\
  RGB images - https://drive.google.com/file/d/1gnlhRghrv7zca8n6vIdUnZTvfjNYIi97/view?usp=share_link
  
  RGB labels - https://drive.google.com/file/d/1gp6rv75MMS3iXvnY2RUsQp7J_7q-D3mQ/view?usp=share_link
  
  Thermal images - https://drive.google.com/file/d/1glhoWTcg2DrIruOrcA2Z90yoDTEXaQtT/view?usp=share_link \\
  Thermal labels - https://drive.google.com/file/d/1gm1PaP_Nlqnrz_u-eKcHOw_R5RKbtpD8/view?usp=share_link \\

# To run the code
1. Replace the path of the datasets (based on the task) in main.py file and test.py file
2. Based on the task change the model specifications as mentioned in the comments. "digit is mentioned for sign digit recognition task i.e. OurModel1, "alpha" is mentioned for alphabet recognition task i.e. OurModel2
