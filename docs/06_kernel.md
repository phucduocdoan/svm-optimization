# 06 — Kernel Trick: Linear và RBF

## 1. Vấn đề dữ liệu phi tuyến

Linear SVM tìm một ranh giới tuyến tính trong không gian đặc trưng hiện tại. Có những dữ liệu không thể chia tốt bằng một đường thẳng, ví dụ XOR.

Kernel Trick thay tích vô hướng `x_i^T x_j` trong Dual bằng một hàm kernel:

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j) \tag{6.1}
$$

Ta không cần tính tường minh vector `phi(x)` có thể nằm trong không gian rất nhiều chiều; chỉ cần tính kernel giữa hai điểm.

![Minh họa Kernel Trick](https://thumb.wikimedia.org/wikipedia/commons/thumb/c/cc/Kernel_trick_idea.svg/1280px-Kernel_trick_idea.svg.png)

Hình minh họa ánh xạ dữ liệu sang không gian đặc trưng để tìm ranh giới tuyến tính. Nguồn: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kernel_trick_idea.svg), Shiyu Ji, CC BY-SA 4.0.

## 2. Linear Kernel

$$
K(x,z)=x^Tz \tag{6.2}
$$

Đây chính là tích vô hướng thông thường. Dùng khi ranh giới tuyến tính là giả định hợp lý hoặc cần baseline dễ diễn giải.

## 3. RBF / Gaussian Kernel

$$
K(x,z)=\exp(-\gamma\|x-z\|^2),\qquad \gamma>0 \tag{6.3}
$$

- Hai điểm gần nhau: kernel gần 1.
- Hai điểm xa nhau: kernel gần 0.
- Gamma lớn: ảnh hưởng của một điểm tập trung trong vùng hẹp; ranh giới linh hoạt hơn và dễ overfit.
- Gamma nhỏ: ảnh hưởng lan rộng; ranh giới mượt hơn.

Cũng phải điều chỉnh `C` cùng gamma bằng validation.

## 4. Hàm quyết định có kernel

Từ `w=sum(alpha_i*y_i*phi(x_i))`, suy ra:

$$
f(x)=\sum_i\alpha_i y_i K(x_i,x)+b \tag{6.4}
$$

Đây là công thức được dùng trong `KernelSVM.decision_function`.

## 5. Liên hệ với code

`src/svm_optimization/models/kernel.py`:

- `linear_kernel(X,Y)` trả `X @ Y.T`.
- `rbf_kernel(X,Y,gamma)` tính khoảng cách bình phương giữa từng cặp rồi lấy `exp(-gamma*d²)`.
- `make_kernel` tạo callable theo tên kernel.

`src/svm_optimization/models/svm.py` dùng cùng một interface để huấn luyện và dự đoán. Khác biệt giữa Linear và RBF chủ yếu nằm ở cách tạo Gram matrix.

## 6. Lưu ý thực nghiệm

- Chuẩn hóa dữ liệu trước khi RBF vì khoảng cách Euclid phụ thuộc scale từng feature.
- `gamma='scale'` trong project được tính theo `1 / (n_features * variance(X_train))`.
- Với RBF, không có vector w hữu hạn trong không gian đầu vào gốc để vẽ trực tiếp như Linear.
- Khi vẽ boundary 2D, hãy huấn luyện lại trên đúng hai feature; không lấy model bốn feature rồi đưa hai feature vào plot.

## 7. Bạn cần tự trả lời

1. Kernel thay thế phần nào trong Dual?
2. Vì sao cần scale dữ liệu trước RBF?
3. RBF gamma lớn có thể làm ranh giới phức tạp hơn theo cách nào?
4. Kernel Trick có nghĩa là ta nhất thiết phải tạo vector phi(x) không?
