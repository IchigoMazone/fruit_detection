from src.utils.progress import Progress
from pydantic import BaseModel, DirectoryPath, ConfigDict, FilePath, Field, validate_call
from typing import Any, Annotated, Literal, TypeAlias
from zipfile import ZipFile
import shutil, os, time, subprocess
import cv2
import numpy as np
import random

def robust_rmtree(path: str, max_retries: int = 5, delay: float = 0.5):
    """
    Robustly removes a directory, retrying if a filesystem lock or 
    'directory not empty' race condition occurs.
    """
    for i in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return
        except Exception as e:
            print(f"Attempt {i+1} to remove {path} failed: {e}")
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                print("Falling back to system rm -rf...")
                try:
                    subprocess.run(["rm", "-rf", path], check=True)
                except Exception as se:
                    print(f"System rm -rf failed: {se}")
                    raise e

CHUNK_SIZE: TypeAlias = Literal[256 * 64, 512 * 64, 1024 * 64]
ClassesType = Annotated[dict[str, Any] | None, Field(init=False)]
NumClassesType = Annotated[int | None, Field(init=False)]

class DataBuilder(BaseModel):

    zip_path: FilePath 
    extract_dir: DirectoryPath
    dataset_dir: DirectoryPath
    objects: dict[str, Any] 
    status: dict[str, Any] 
    classes: ClassesType = None
    num_classes: NumClassesType = None

    @validate_call
    def extract_zip(
        self, 
        progress: bool = True, 
        bar_format: str | None = None, 
        ascii: str | bool | None = None, 
        unit: str | None = None, 
        chunk_size: CHUNK_SIZE = 1024 * 64, 
        **kwargs
    ):

        progress = Progress(**kwargs)
        with ZipFile(self.zip_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            total_size = sum(f.file_size for f in file_list if not f.is_dir())

            tracker = progress.start(total_size, bar_format, ascii, unit) if progress else None
            for file in file_list:

                if file.is_dir() or ".." in file.filename:
                    continue

                output_path = os.path.join(self.extract_dir, file.filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with zip_ref.open(file) as source, open(output_path, "wb") as target:
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break

                        target.write(chunk)
                        if tracker:
                            tracker.update(len(chunk))



    @validate_call
    def get_fruit_bbox(self, img: Any) -> tuple[int, int, int, int]:
        h_orig, w_orig, _ = img.shape
        
        # Downsample for speed
        img_small = cv2.resize(img, (128, 128))
        h, w, _ = img_small.shape
        pixels = img_small.reshape(-1, 3).astype(np.float32)
        
        # K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape(h, w)
        
        # Compute edge counts for each cluster
        edge_counts = np.zeros(K)
        for k in range(K):
            edge_pixels = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
            edge_counts[k] = np.sum(edge_pixels == k)
            
        # Classify clusters to identify background
        valid_clusters = []
        for k in range(K):
            b, g, r = centers[k]
            val = max(b, g, r)
            sat = (val - min(b, g, r)) / val if val > 0 else 0
            
            is_black = val < 50
            is_white = val > 150 and sat < 0.25
            
            if not is_black and not is_white:
                valid_clusters.append(k)
                
        if valid_clusters:
            fruit_cluster = min(valid_clusters, key=lambda k: edge_counts[k])
        else:
            fruit_cluster = np.argmin(edge_counts)
            
        mask = (labels == fruit_cluster).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Resize mask back to original size
        mask_large = cv2.resize(opened, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        contours, _ = cv2.findContours(mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
            
        return (0, 0, w_orig, h_orig)

    @validate_call
    def build_yolo_dataset(
        self, 
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1), 
        seed: int = 42, 
        max_images_per_category: int | None = 667
    ):
        # 1. Ensure the outer zip is extracted
        image_dir = os.path.join(self.extract_dir, "Augmented-Resized Image")
        if not os.path.exists(image_dir):
            print(f"Directory {image_dir} not found. Extracting outer zip...")
            self.extract_zip()
            
        # 2. Setup YOLO dataset directories
        yolo_dir = os.path.join(self.dataset_dir, "yolo_dataset")
        if os.path.exists(yolo_dir):
            print(f"Removing existing YOLO dataset directory at {yolo_dir}...")
            robust_rmtree(yolo_dir)
            
        images_dir = os.path.join(yolo_dir, "images")
        labels_dir = os.path.join(yolo_dir, "labels")
        
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
            
        temp_dir = os.path.join(self.extract_dir, "temp_extracted")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 3. Find and count all images across inner zip files
        inner_zips = []
        total_images = 0
        
        for fruit in self.objects.keys():
            fruit_dir = os.path.join(image_dir, fruit)
            if not os.path.exists(fruit_dir):
                print(f"Warning: Fruit directory {fruit_dir} does not exist.")
                continue
            for state in self.status.keys():
                zip_path = os.path.join(fruit_dir, f"{state}.zip")
                if os.path.exists(zip_path):
                    with ZipFile(zip_path, "r") as z:
                        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
                        count = len(names)
                        if max_images_per_category is not None:
                            count = min(count, max_images_per_category)
                        total_images += count
                        inner_zips.append((fruit, state, zip_path, names))
                else:
                    print(f"Warning: Zip file {zip_path} does not exist.")
                    
        print(f"Total images found to process: {total_images}")
        if total_images == 0:
            print("No images found. Exiting.")
            return
            
        # 4. Process images
        random.seed(seed)
        
        progress = Progress(desc="Building YOLO Dataset", unit="img")
        tracker = progress.start(total=total_images)
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        # Normalize split ratios to sum to 1.0
        r_train, r_val, r_test = split_ratio
        total_ratio = r_train + r_val + r_test
        r_train /= total_ratio
        r_val /= total_ratio
        r_test /= total_ratio
        
        for fruit, state, zip_path, image_names in inner_zips:
            with ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)
                
            random.shuffle(image_names)
            if max_images_per_category is not None:
                image_names = image_names[:max_images_per_category]
                
            n_total = len(image_names)
            idx_train = int(n_total * r_train)
            idx_val = int(n_total * (r_train + r_val))
            
            train_names = image_names[:idx_train]
            val_names = image_names[idx_train:idx_val]
            test_names = image_names[idx_val:]
            
            for split_name, name_list in [("train", train_names), ("val", val_names), ("test", test_names)]:
                dest_img_dir = os.path.join(images_dir, split_name)
                dest_lbl_dir = os.path.join(labels_dir, split_name)
                
                for name in name_list:
                    src_path = os.path.join(temp_dir, name)
                    if not os.path.exists(src_path):
                        src_path_alt = os.path.join(temp_dir, os.path.basename(name))
                        if os.path.exists(src_path_alt):
                            src_path = src_path_alt
                        else:
                            tracker.update(1)
                            continue
                            
                    img = cv2.imread(src_path)
                    if img is not None:
                        x, y, w_box, h_box = self.get_fruit_bbox(img)
                        
                        img_h, img_w, _ = img.shape
                        x_center = (x + w_box / 2.0) / img_w
                        y_center = (y + h_box / 2.0) / img_h
                        yolo_w = w_box / img_w
                        yolo_h = h_box / img_h
                        
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        yolo_w = min(max(yolo_w, 0.0), 1.0)
                        yolo_h = min(max(yolo_h, 0.0), 1.0)
                        
                        base_name = os.path.splitext(os.path.basename(name))[0]
                        new_base_name = f"{fruit}_{state}_{base_name}"
                        dest_img_path = os.path.join(dest_img_dir, f"{new_base_name}.jpg")
                        dest_lbl_path = os.path.join(dest_lbl_dir, f"{new_base_name}.txt")
                        
                        shutil.copy(src_path, dest_img_path)
                        
                        with open(dest_lbl_path, "w") as f_lbl:
                            f_lbl.write(f"0 {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                            
                        if split_name == "train":
                            train_count += 1
                        elif split_name == "val":
                            val_count += 1
                        elif split_name == "test":
                            test_count += 1
                            
                    tracker.update(1)
                    
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    robust_rmtree(item_path)
                else:
                    try:
                        os.remove(item_path)
                    except Exception as e:
                        print(f"Warning: could not remove file {item_path}: {e}")
                    
        if os.path.exists(temp_dir):
            robust_rmtree(temp_dir)
            
        tracker.finish(end=True)
        
        yolo_abs_path = os.path.abspath(yolo_dir)
        yaml_content = f"path: {yolo_abs_path}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 1\nnames:\n  0: fruit\n"
        yaml_path = os.path.join(yolo_dir, "dataset.yaml")
        with open(yaml_path, "w") as f_yaml:
            f_yaml.write(yaml_content)
            
        print("\nDataset generation summary:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
        print(f"  Testing images: {test_count}")
        print(f"  Total processed: {train_count + val_count + test_count}")
        print(f"  YOLO config saved to: {yaml_path}")

    @validate_call
    def extract_all(self):
        self.build_yolo_dataset()


if __name__ == "__main__":

    builder = DataBuilder(**{
        "zip_path": "data/Augmented-Resized Image.zip",
        "extract_dir": "data/",
        "dataset_dir": "data/",
        "objects": {
            "Apple": "Apple",
            "Banana": "Banana",
            "Grape": "Grape",
            "Mango": "Mango",
            "Orange": "Orange"
        },
        "status": {
            "Formalin-mixed": "Formalin",
            "Fresh": "Fresh",
            "Rotten": "Rotten"
        }
    })

    builder.build_yolo_dataset(max_images_per_category=667)