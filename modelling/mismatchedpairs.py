import os

def find_missing_pairs(image_folder, label_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(label_folder) if f.endswith(".png")]

    image_numbers = set(int(f.split("_")[1].split(".")[0]) for f in image_files)
    label_numbers = set(int(f.split("_")[1].split(".")[0]) for f in label_files)

    missing_numbers = image_numbers.symmetric_difference(label_numbers)

    print("Missing image files:")
    for num in missing_numbers:
        if num not in image_numbers:
            print(f"image_%d.jpg" % num)

    print("Missing label files:")
    for num in missing_numbers:
        if num not in label_numbers:
            print(f"anno_%d.png" % num)

image_folder = "D:/Storage/Random_Stuff/Stanl/DigitizationWork/AFSCRunningModel/dev_workflow/example/samples_reformatted_0.075m_512x512_lanczos/image"
label_folder = "D:/Storage/Random_Stuff/Stanl/DigitizationWork/AFSCRunningModel/dev_workflow/example/samples_reformatted_0.075m_512x512_lanczos/label"
find_missing_pairs(image_folder, label_folder)
