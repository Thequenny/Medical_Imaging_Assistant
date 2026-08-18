import numpy as np
import os
import os.path 
from pathlib import Path
import shutil
import dicom2nifti
from pydicom import dcmread

# To recover the path to the data
project_dir=os.path.dirname(os.path.abspath(__file__))

data_CT_dir=os.path.join(project_dir,"data_base_CT")
data_Nii_dir=os.path.join(project_dir,"data_base_nii")
buffer=os.path.join(project_dir,"temp")

def CTdcm_to_Nii(CT_dir, output_buffer, nii_dir):

    CT_dir = Path(CT_dir)
    output_buffer = Path(output_buffer)
    nii_dir = Path(nii_dir)

    non_dcm_dir = CT_dir/"non_dicom"
    os.makedirs(non_dcm_dir, exist_ok=True)
    #non_dcm_dir.mkdir(exist_ok=True)
    case = CT_dir.name
    case_buffer = output_buffer/case
    has_dcm = False
    dcm_files = []
    if case_buffer.exists():
        shutil.rmtree(case_buffer)
    case_buffer.mkdir(parents=True, exist_ok=True)

    # Sort the folder to keep only the .dcm files 
    for element in CT_dir.iterdir():
        if element.is_file(): 
        
            if element.suffix.lower() == ".dcm":
                has_dcm = True
                dcm_files.append(element)
            else:
                print("the file ",element," is not a .dcm file")
                shutil.move(str(element), str(non_dcm_dir/element.name))
                print("no .dcm file moved")

        elif element.is_dir() and element.name != "non_dicom":
                CTdcm_to_Nii(element,output_buffer, nii_dir)

    if has_dcm:
        first_dcm = dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
        series_description = str(getattr(first_dcm, "SeriesDescription", "")).lower()
        image_type = " ".join(str(value).lower() for value in getattr(first_dcm, "ImageType", []))

        if len(dcm_files) < 4 or "scout" in series_description or "localizer" in image_type:
            print("Conversion skipped for", CT_dir)
            print("Reason: too few slices or scout/localizer series")
            shutil.rmtree(case_buffer)
            return

        dicom2nifti.convert_directory(
            str(CT_dir),
            str(case_buffer),
            compression=True,
            reorient=True
        )
    print(case_buffer.exists())
    print(list(case_buffer.iterdir()))       
    buffer_files = os.listdir(case_buffer)

    for b in buffer_files:
        buffer_path = os.path.join(case_buffer, b)

        if not (b.endswith(".nii") or b.endswith(".nii.gz")):
            continue

        niisave_path = os.path.join(nii_dir, case + ".nii.gz")

        print("from:", buffer_path)
        print("to:", niisave_path)
            
        shutil.move(buffer_path, niisave_path)
        shutil.rmtree(case_buffer)
        break
    

if __name__ == "__main__":
    os.makedirs(data_Nii_dir, exist_ok=True)
    os.makedirs(buffer, exist_ok=True)
    CTdcm_to_Nii(data_CT_dir, buffer, data_Nii_dir)
