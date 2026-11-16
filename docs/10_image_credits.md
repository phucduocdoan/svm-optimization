# 10 — Nguồn dữ liệu và cách tạo hình

Các hình khoa học trong `docs/images/scientific/` được vẽ bằng Matplotlib từ dữ liệu tổng hợp hoặc công thức, không phải kết quả Banknote. Tạo lại PNG và SVG từ thư mục gốc của project:

```bash
uv run python docs/images/scientific/generate.py
```

Bốn sơ đồ trong `docs/diagrams/` có nguồn Mermaid `.mmd`; Markdown chứa cùng nội dung để GitHub render trực tiếp. Để tạo lại PNG dùng cho báo cáo Word, cài Playwright và Chromium, sau đó chạy:

```bash
uv run --with playwright python -m playwright install chromium
uv run --with playwright python docs/diagrams/render.py
```

Bước này cần mạng để tải thư viện Mermaid. Có thể đặt `CHROMIUM_EXECUTABLE_PATH` để dùng Chromium đã cài. Khi sửa sơ đồ, cập nhật cả nguồn `.mmd` và khối Mermaid tương ứng trong Markdown, sau đó tạo lại PNG và báo cáo Word.

Các biểu đồ kết quả trong `reports/figures/` được sinh từ thí nghiệm của project. Không dùng chúng làm kết quả cho một cấu hình khác.

## Dataset

- Banknote Authentication: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication), Volker Lohweg (2012), DOI [10.24432/C55P57](https://doi.org/10.24432/C55P57), giấy phép CC BY 4.0.

## Tài liệu tham khảo bổ sung

- Phạm Đình Khanh, *Machine Learning Algorithms to Practice*, chương 7 "Support Vector Machine", <https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html> (truy cập 2026-11-16). Tham khảo ý tưởng giải thích cho phần trực giác "công bằng" của biên ([01](01_svm_geometry.md)), ba trường hợp của $\xi_i$ ([02](02_primal_soft_margin.md)), ví dụ QP nhỏ với nhân tử Lagrange ([03](03_lagrange_dual.md)), kernel đa thức bậc 2 qua ánh xạ đặc trưng tường minh và trực giác landmark của RBF ([06](06_kernel.md)). Nội dung và ví dụ được viết lại bằng lời của nhóm; không sao chép văn bản hay hình ảnh từ sách.
