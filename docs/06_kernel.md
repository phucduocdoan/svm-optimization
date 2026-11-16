# 06 — Kernel Trick: Linear và RBF

## 1. Vấn đề dữ liệu phi tuyến

Linear SVM tìm một ranh giới tuyến tính trong không gian đặc trưng hiện tại. Có những dữ liệu không thể chia tốt bằng một đường thẳng, ví dụ XOR.

Kernel Trick thay tích vô hướng $x_i^T x_j$ trong Dual bằng một hàm kernel:

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)\qquad\text{(6.1)}
$$

Ta không cần tính tường minh vector $\phi(x)$ có thể nằm trong không gian rất nhiều chiều; chỉ cần tính kernel giữa hai điểm.

![Ranh giới Linear và RBF trên dữ liệu XOR tổng hợp](images/scientific/linear_rbf_xor.svg)

*Hình 4: Hai mô hình SVC học cùng dữ liệu XOR tổng hợp. Đây là ví dụ trực quan trên tập huấn luyện, không phải benchmark Banknote.*

## 2. Linear Kernel

$$
K(x,z)=x^Tz\qquad\text{(6.2)}
$$

Đây chính là tích vô hướng thông thường. Dùng khi ranh giới tuyến tính là giả định hợp lý hoặc cần baseline dễ diễn giải.

### Kernel đa thức bậc 2 chính là một $\phi$ tường minh

Công thức (6.1) nói $K(x,z)=\phi(x)^T\phi(z)$ cho một $\phi$ nào đó, nhưng không bắt phải biết $\phi$ là gì. Với $x=(x_1,x_2)\in\mathbb{R}^2$, kernel đa thức bậc 2 $K(x,z)=(x^Tz)^2$ là một trường hợp hiếm có thể viết $\phi$ ra tường minh:

$$
\phi(x)=(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2).
$$

Khai triển $(x^Tz)^2=(x_1z_1+x_2z_2)^2=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2$, đúng bằng $\phi(x)^T\phi(z)$.

Thử với số cụ thể $x=(1,2)$, $z=(3,4)$: $x^Tz=1\times3+2\times4=11$ nên $K(x,z)=11^2=121$. Mặt khác $\phi(x)=(1,\ 2\sqrt2,\ 4)$ và $\phi(z)=(9,\ 12\sqrt2,\ 16)$, tích vô hướng $\phi(x)^T\phi(z)=1\times9+2\sqrt2\times12\sqrt2+4\times16=9+48+64=121$, khớp với $K(x,z)$ tính trực tiếp. Đây chính là lý do kernel trick hữu ích: ta chỉ cần tính $(x^Tz)^2$, không cần dựng $\phi(x)$ ba chiều.

*Ý tưởng minh họa tham khảo từ Phạm Đình Khanh, Machine Learning Algorithms to Practice, ch. 7 SVM: [phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html](https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html).*

## 3. RBF / Gaussian Kernel

RBF có thể hiểu như một phép đo "giống nhau" giữa $x$ và một điểm mốc (landmark) $l$: nếu $x$ rất gần $l$, giá trị kernel gần $1$; nếu $x$ ở xa $l$, giá trị gần $0$. Ánh xạ một điểm sang các giá trị giống-nhau với vài landmark như vậy có thể biến dữ liệu kiểu XOR — vốn không tách được bằng một đường thẳng trong không gian gốc — thành dữ liệu tách được tuyến tính trong không gian mới.

Ví dụ số trên bốn đỉnh XOR $(0,0),(1,1)$ thuộc lớp $-1$ và $(0,1),(1,0)$ thuộc lớp $+1$: chọn hai landmark $l_1=(0,0)$, $l_2=(1,1)$ và $\gamma=1$. Với $f_1=K(x,l_1)$, $f_2=K(x,l_2)$:

| $x$ | lớp | $f_1$ | $f_2$ | $f_1+f_2$ |
|---|---|---:|---:|---:|
| $(0,0)$ | $-1$ | $1$ | $0.135$ | $1.135$ |
| $(1,1)$ | $-1$ | $0.135$ | $1$ | $1.135$ |
| $(0,1)$ | $+1$ | $0.368$ | $0.368$ | $0.736$ |
| $(1,0)$ | $+1$ | $0.368$ | $0.368$ | $0.736$ |

Hai điểm lớp $-1$ có $f_1+f_2\approx1.135$, hai điểm lớp $+1$ có $f_1+f_2\approx0.736$: một ngưỡng đơn giản như $f_1+f_2=0.9$ đã tách được hai lớp, dù trong không gian $(x_1,x_2)$ gốc XOR không tách được bằng đường thẳng nào.

*Ý tưởng minh họa tham khảo từ Phạm Đình Khanh, Machine Learning Algorithms to Practice, ch. 7 SVM: [phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html](https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html).*

$$
K(x,z)=\exp(-\gamma\|x-z\|^2),\qquad \gamma>0\qquad\text{(6.3)}
$$

- Hai điểm gần nhau: kernel gần 1.
- Hai điểm xa nhau: kernel gần 0.
- Gamma lớn: ảnh hưởng của một điểm tập trung trong vùng hẹp; ranh giới linh hoạt hơn và dễ overfit.
- Gamma nhỏ: ảnh hưởng lan rộng; ranh giới mượt hơn.

Cũng phải điều chỉnh $C$ cùng gamma bằng validation.

## 4. Hàm quyết định có kernel

Từ $w=\sum_i\alpha_i y_i\phi(x_i)$, suy ra:

$$
f(x)=\sum_i\alpha_i y_i K(x_i,x)+b\qquad\text{(6.4)}
$$

Đây là công thức được dùng trong `KernelSVM.decision_function`.

## Ví dụ số: từ kernel đến dự đoán

Lấy hai mẫu một chiều $x_1=-1$, $x_2=1$, nhãn $y_1=-1$, $y_2=1$ và RBF với $\gamma=0.5$. Phần tử hàng $i$, cột $j$ của Gram matrix là $K(x_i,x_j)$:

| $K(x_i,x_j)$ | $x_1=-1$ | $x_2=1$ |
|---|---:|---:|
| $x_1=-1$ | $1$ | $e^{-2}\approx0.1353$ |
| $x_2=1$ | $e^{-2}\approx0.1353$ | $1$ |

Với điểm mới $x=0.5$, ta tính $K(x_1,x)\approx0.3247$ và $K(x_2,x)\approx0.8825$. Trên ví dụ này với $C=1$, mô hình học được $\alpha_1=\alpha_2=1$ và $b\approx0$. Thay vào công thức (6.4):

$$
f(x)\approx (-1)(0.3247)+(1)(0.8825)+0=0.5578.
$$

Score dương nên mô hình dự đoán lớp $+1$. Các số trên dùng mẫu rất nhỏ để học cách tính, không phải kết quả Banknote. Chạy ví dụ để xem cả Gram matrix Linear, từng số hạng và score từ `KernelSVM`:

```bash
uv run python -m experiments.07_kernel_trick
```

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
