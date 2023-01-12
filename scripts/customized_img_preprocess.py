from PIL import Image
import os

IMG_ROOT = "/home/local/ASUAD/weiweigu/Downloads/temp"
TARGET_DIR = "/home/local/ASUAD/weiweigu/data/customized_set/images"
# number of imgs before adding
STRT_IDX = 11

if __name__ == "__main__":
    img_list = os.listdir(IMG_ROOT)
    for i, img_file_name in enumerate(img_list):
        input_path = os.path.join(IMG_ROOT,f"{img_file_name}")
        img = Image.open(input_path)
        target_img = img.resize((224,224))
        output_path = os.path.join(TARGET_DIR, f"{i+STRT_IDX}.jpg")
        target_img.save(output_path)