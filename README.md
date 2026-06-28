# Fruit Detection App

Ứng dụng phát hiện trái cây bằng YOLO và chế độ YOLO + ResNet.

## Yêu cầu

- Python 3.12 trở lên
- uv để cài thư viện Python
- Node.js LTS và npm để chạy frontend

Tải Node.js tại:

```text
https://nodejs.org/
```

Kiểm tra phiên bản:

```bash
python3 --version
uv --version
node -v
npm -v
```

## Cài uv

Nếu chưa có `uv`, cài bằng lệnh:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Đóng terminal và mở lại, sau đó kiểm tra:

```bash
uv --version
```

## Cài thư viện backend

Tại thư mục project:

```bash
cd /home/nhattrinh/Downloads/MLP
uv sync
```

Lệnh này sẽ tạo `.venv` và cài các thư viện Python trong `pyproject.toml`, gồm FastAPI, Uvicorn, Ultralytics, PyTorch CUDA, OpenCV.

## Cài thư viện frontend

```bash
cd /home/nhattrinh/Downloads/MLP/frontend
npm install
```

## Chạy backend

Mở terminal tại thư mục project:

```bash
cd /home/nhattrinh/Downloads/MLP
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Backend chạy tại:

```text
http://127.0.0.1:8000
```

## Chạy frontend

Mở terminal khác:

```bash
cd /home/nhattrinh/Downloads/MLP/frontend
npm run dev
```

Frontend chạy tại:

```text
http://localhost:3000
```
