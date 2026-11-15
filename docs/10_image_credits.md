# 10 — Nguồn dữ liệu và cách tạo hình

Các hình khoa học trong `docs/images/scientific/` được vẽ bằng Matplotlib từ dữ liệu tổng hợp hoặc công thức, không phải kết quả Banknote. Tạo lại PNG và SVG từ thư mục gốc của project:

```bash
python docs/images/scientific/generate.py
```

Bốn sơ đồ trong `docs/diagrams/` có nguồn Mermaid `.mmd`; Markdown chứa cùng nội dung để GitHub render trực tiếp. Để tạo lại PNG dùng cho báo cáo Word, cài Playwright và Chromium, sau đó chạy:

```bash
python -m pip install playwright
python -m playwright install chromium
python docs/diagrams/render.py
```

Bước này cần mạng để tải thư viện Mermaid. Có thể đặt `CHROMIUM_EXECUTABLE_PATH` để dùng Chromium đã cài. Khi sửa sơ đồ, cập nhật cả nguồn `.mmd` và khối Mermaid tương ứng trong Markdown, sau đó tạo lại PNG và báo cáo Word.

Các biểu đồ kết quả trong `reports/figures/` được sinh từ thí nghiệm của project. Không dùng chúng làm kết quả cho một cấu hình khác.

## Dataset

- Banknote Authentication: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication), Volker Lohweg (2012), DOI [10.24432/C55P57](https://doi.org/10.24432/C55P57), giấy phép CC BY 4.0.
