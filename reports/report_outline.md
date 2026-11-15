# Đề cương báo cáo đồ án — Cơ sở toán học và phương pháp tối ưu SVM

## Trang bìa
Tên trường, học phần Tối ưu hóa, tên đề tài, giảng viên, sinh viên, thời gian.

## Tóm tắt
Nêu bài toán, SVM nhị phân, phương pháp tối ưu được nghiên cứu (Primal/Dual/SMO), dataset và kết quả chính.

## Chương 1. Giới thiệu
- Bài toán phân loại nhị phân.
- Mục tiêu và phạm vi đề tài.
- Vì sao SVM là bài toán tối ưu.

## Chương 2. Cơ sở toán học
- Vector, tích vô hướng, norm.
- Hyperplane, khoảng cách, margin.
- Hard-margin và soft-margin.
- Primal optimization problem.
- Lagrangian và suy ra Dual.
- KKT và ý nghĩa support vectors.

## Chương 3. Thuật toán giải
- Quadratic programming.
- SMO: chọn cặp, tính L/H, cập nhật alpha, cập nhật bias.
- Linear và RBF kernel.
- Sơ đồ thuật toán.

## Chương 4. Cài đặt
- Kiến trúc project và class.
- Module toán.
- SMOOptimizer và KernelSVM.
- Baseline scikit-learn.
- Kiểm thử ví dụ giải tay.

## Chương 5. Thực nghiệm
- Banknote Authentication.
- Tiền xử lý và chia dữ liệu.
- Kết quả baseline/custom.
- So sánh C và kernel.
- Accuracy, Precision, Recall, F1, Confusion Matrix.
- Objective, dual feasibility, support vectors, convergence.

## Chương 6. Thảo luận
- Ý nghĩa tối ưu hóa của kết quả.
- Ưu/nhược điểm SMO bản học tập.
- Hạn chế dữ liệu và thực nghiệm.

## Kết luận và hướng phát triển

## Tài liệu tham khảo

## Phụ lục
Code, hướng dẫn chạy, kết quả tests và bảng thực nghiệm.
