import os
import torch
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime

# Cấu hình PyTorch để tiết kiệm bộ nhớ
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Đường dẫn đến file cấu hình dữ liệu
data_yaml = '/home/nvidia/iDragonCloud/Bone_Fracture_Detection/BoneFractureYolo8/data.yaml'

# Tạo thư mục để lưu kết quả
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'/home/nvidia/iDragonCloud/Bone_Fracture_Detection/results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

# Tạo và huấn luyện model
def train_model():
    print("=== Bắt đầu huấn luyện mô hình YOLOv8 cho phát hiện gãy xương ===")
    
    # Tạo model từ pretrained weights
    model = YOLO('yolov8n.pt')  # Sử dụng mô hình nano để tiết kiệm bộ nhớ
    
    # Cấu hình huấn luyện tiết kiệm bộ nhớ
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=640,        # Giảm kích thước ảnh
        batch=16,          # Giảm batch size
        workers=0,        # Tắt multiprocessing loading
        cache=True,      # Không sử dụng cache
        device=0,         # Sử dụng GPU (0: GPU đầu tiên)
        patience=30,      # Early stopping
        project=results_dir,
        name='train',
    )
    
    # Trả về model đã huấn luyện và kết quả
    return model, results

# Hàm đánh giá mô hình và hiển thị các thông số
def evaluate_model(model):
    print("\n=== Đánh giá mô hình trên tập validation ===")
    
    # Đánh giá trên tập validation
    val_results = model.val(data=data_yaml, split='val')
    
    # Hiển thị và lưu các thông số đánh giá
    metrics = {
        'mAP50': val_results.box.map50,
        'mAP50-95': val_results.box.map,
        'Precision': val_results.box.mp,
        'Recall': val_results.box.mr,
        'F1-Score': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr + 1e-16)
    }
    
    print("\n--- Metrics trên tập validation ---")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Lưu metrics ra file
    with open(f"{results_dir}/metrics.txt", "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    # Visualize các metrics trong plots
    plot_metrics(metrics)
    
    return val_results

# Hàm trực quan hóa các metrics
def plot_metrics(metrics):
    print("\n=== Tạo biểu đồ hiển thị metrics ===")
    
    # Vẽ biểu đồ cột cho các metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylim(0, 1.0)
    plt.title('YOLOv8 Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Thêm giá trị lên đầu các cột
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        plt.text(i, metric_value + 0.02, f'{metric_value:.4f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics_visualization.png")
    print(f"Đã lưu biểu đồ metrics tại: {results_dir}/metrics_visualization.png")

# Hiển thị dự đoán trên một số ảnh test
def visualize_predictions(model):
    print("\n=== Hiển thị dự đoán trên ảnh test ===")
    
    # Đọc thông tin từ data.yaml để lấy đường dẫn đến tập test
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    base_path = data_config.get('path', '')
    test_path = os.path.join(base_path, data_config.get('test', 'test/images'))
    
    # Đảm bảo đường dẫn test tồn tại
    if not os.path.exists(test_path):
        print(f"Không tìm thấy thư mục test: {test_path}")
        # Sử dụng thư mục val nếu không có test
        test_path = os.path.join(base_path, data_config.get('val', 'val/images'))
        print(f"Sử dụng thư mục validation thay thế: {test_path}")
    
    # Lấy danh sách ảnh trong thư mục test
    test_images = [os.path.join(test_path, img) for img in os.listdir(test_path)
                  if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Chọn 10 ảnh ngẫu nhiên để dự đoán (hoặc ít hơn nếu không đủ)
    import random
    sample_size = min(10, len(test_images))
    sample_images = random.sample(test_images, sample_size)
    
    # Dự đoán trên ảnh mẫu
    for img_path in sample_images:
        results = model.predict(
            source=img_path,
            conf=0.25,  # Ngưỡng confidence
            save=True,  # Lưu kết quả
            project=results_dir,
            name='predictions',
            exist_ok=True
        )
    
    print(f"Đã lưu các ảnh dự đoán tại: {results_dir}/predictions")

# Hàm chạy toàn bộ quy trình
def main():
    try:
        # Huấn luyện mô hình
        model, train_results = train_model()
        
        # Đánh giá mô hình
        val_results = evaluate_model(model)
        
        # Hiển thị dự đoán trên ảnh test
        visualize_predictions(model)
        
        # Xuất mô hình
        model.export(format="onnx", imgsz=640)
        print(f"\n=== Đã xuất mô hình sang định dạng ONNX tại: {results_dir}/train/weights/best.onnx ===")
        
        print("\n=== Toàn bộ quá trình huấn luyện và đánh giá đã hoàn tất ===")
        print(f"Kết quả được lưu tại thư mục: {results_dir}")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()