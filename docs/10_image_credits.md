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

## Tài liệu tham khảo

Các trích dẫn [n] trong docs trỏ tới danh sách này.

1. <a id="ref-1"></a>Cortes, C. & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20, 273–297.
2. <a id="ref-2"></a>Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. MSR-TR-98-14.
3. <a id="ref-3"></a>Chang, C.-C. & Lin, C.-J. (2011). LIBSVM: A Library for Support Vector Machines.
4. <a id="ref-4"></a>UCI Banknote Authentication: <https://archive.ics.uci.edu/dataset/267/banknote+authentication>
5. <a id="ref-5"></a>Phạm Đình Khanh. *Machine Learning Algorithms to Practice*, chương 7 "Support Vector Machine". <https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html> (truy cập 2026-11-16). Chỉ tham khảo ý tưởng giải thích; văn bản, dữ liệu tổng hợp, hình và code trong docs do nhóm tự viết.
