# 03 — Lagrangian và bài toán đối ngẫu (Dual)

Đây là phần toán trọng tâm của đồ án. Hãy hiểu ý nghĩa trước, sau đó tự viết lại từng phép đạo hàm.

![Gradient descent trên đường đồng mức](https://commons.wikimedia.org/wiki/Special:FilePath/Gradient_descent.png?width=1000)

Hình minh họa trực giác chung về đi xuống theo hướng giảm của một hàm mục tiêu; Dual của SVM được giải bằng SMO chứ không phải gradient descent thông thường. Nguồn: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent.png), public domain.

## 1. Nhắc lại Primal

$$
\begin{aligned}
\min_{w,b,\xi}\quad &\frac12\|w\|^2+C\sum_i\xi_i\\
\text{s.t.}\quad &1-y_i(w^Tx_i+b)-\xi_i\leq0,\\
&-\xi_i\leq0.
\end{aligned} \tag{3.1}
$$

Ta đưa mỗi ràng buộc bất đẳng thức vào Lagrangian bằng một biến nhân tử không âm:

- `alpha_i >= 0` cho ràng buộc margin.
- `mu_i >= 0` cho ràng buộc `xi_i >= 0`.

## 2. Lagrangian

$$
\begin{aligned}
L(w,b,\xi,\alpha,\mu)
=&\frac12\|w\|^2+C\sum_i\xi_i\\
&-\sum_i\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]\\
&-\sum_i\mu_i\xi_i.
\end{aligned} \tag{3.2}
$$

Bạn có thể đọc từng dòng:

- Dòng đầu là objective gốc.
- Dòng hai gắn điều kiện phân loại vào Lagrangian.
- Dòng ba gắn điều kiện slack không âm.

## 3. Lấy đạo hàm riêng và cho bằng 0

### Theo w

$$
\frac{\partial L}{\partial w}=w-\sum_i\alpha_i y_i x_i=0
$$

Suy ra:

$$
\boxed{w=\sum_i\alpha_i y_i x_i} \tag{3.3}
$$

Ý nghĩa: vector pháp tuyến tối ưu là tổ hợp có trọng số của các mẫu dữ liệu. Mẫu có alpha bằng 0 không đóng góp vào w.

### Theo b

$$
\frac{\partial L}{\partial b}=-\sum_i\alpha_i y_i=0
$$

Suy ra:

$$
\boxed{\sum_i\alpha_i y_i=0} \tag{3.4}
$$

Đây là ràng buộc đẳng thức của bài toán Dual.

### Theo xi_i

$$
\frac{\partial L}{\partial\xi_i}=C-\alpha_i-\mu_i=0
$$

Vì `mu_i>=0` và `alpha_i>=0`:

$$
\mu_i=C-\alpha_i\quad\Longrightarrow\quad 0\leq\alpha_i\leq C \tag{3.5}
$$

Đây là nguồn gốc của cận trên C trong bài toán đối ngẫu, không phải một điều kiện được đặt tùy ý.

## 4. Thay trở lại Lagrangian

Sau khi loại `w,b,xi,mu`, ta nhận được:

$$
\boxed{
\begin{aligned}
\max_{\alpha}\quad &\sum_i\alpha_i-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i^Tx_j)\\
\text{s.t.}\quad &\sum_i\alpha_i y_i=0,\\
&0\leq\alpha_i\leq C.
\end{aligned}} \tag{3.6}
$$

Đây là bài toán đối ngẫu bậc hai. Ta tối đa hóa một hàm lõm theo alpha (tương đương tối thiểu hóa hàm lồi sau khi đổi dấu), với các ràng buộc tuyến tính.

## 5. Primal và Dual khác nhau thế nào?

| Primal | Dual |
|---|---|
| Biến chính `w,b,xi` | Biến chính `alpha` |
| Tối thiểu hóa | Tối đa hóa |
| Ràng buộc trên margin và slack | Tổng alpha*y bằng 0; alpha nằm trong [0,C] |
| Dễ diễn giải hình học | Là nền tảng cho Support Vectors và Kernel Trick |

Theo điều kiện Slater của bài toán lồi này, strong duality được đảm bảo trong điều kiện chuẩn của Soft-Margin SVM: nghiệm tối ưu Primal và Dual có cùng giá trị mục tiêu. Điều này cho phép dùng khoảng cách primal-dual để kiểm tra nghiệm số.

## 6. Ví dụ kiểm tra ràng buộc Dual

Với `y=[-1,+1,-1,+1]` và `alpha=[0.25,0.25,0,0]`:

$$
\sum_i\alpha_i y_i=0.25(-1)+0.25(+1)=0.
$$

Nếu `C=1`, từng alpha nằm trong `[0,1]`, nên nghiệm thỏa ràng buộc Dual.

## 7. Liên hệ với code

File `src/svm_optimization/math/dual.py`:

- `dual_objective(alpha,y,K)`: tính hàm mục tiêu Dual.
- `dual_gradient(...)`: tính gradient theo alpha.
- `check_dual_feasibility(...)`: kiểm tra cận alpha và ràng buộc equality.

`K` là Gram matrix, với `K[i,j] = x_i^T x_j` ở Linear Kernel. Khi dùng kernel, phép tích vô hướng được thay bằng `K(x_i,x_j)`.

## 8. Bạn cần tự trả lời

1. `alpha_i` xuất hiện từ ràng buộc nào?
2. Vì sao `alpha_i<=C`?
3. Ràng buộc `sum(alpha_i*y_i)=0` xuất hiện từ đạo hàm theo biến nào?
4. Vì sao Dual thuận tiện để đưa Kernel vào?
