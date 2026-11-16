# 03 — Lagrangian và bài toán đối ngẫu (Dual)

Đây là phần toán trọng tâm của đồ án. Hãy hiểu ý nghĩa trước, sau đó tự viết lại từng phép đạo hàm.

## 0. Khởi động: một bài QP tí hon với một nhân tử

Trước khi vào Lagrangian của SVM, hãy giải một bài toán quy hoạch bậc hai (QP) rất nhỏ: hai biến, một ràng buộc tuyến tính, để thấy nhân tử Lagrange hoạt động ra sao trong trường hợp đơn giản nhất.

Xét bài toán

$$
\min_{x,y}\ x^2+2y^2\quad\text{sao cho}\quad x+y=3.
$$

Gắn ràng buộc vào hàm mục tiêu bằng một nhân tử $\lambda$:

$$
L(x,y,\lambda)=x^2+2y^2-\lambda(x+y-3).
$$

Cho đạo hàm riêng bằng 0:

$$
\frac{\partial L}{\partial x}=2x-\lambda=0,\qquad\frac{\partial L}{\partial y}=4y-\lambda=0
$$

nên $x=\lambda/2$ và $y=\lambda/4$. Thay vào ràng buộc $x+y=3$: $\lambda/2+\lambda/4=3$, suy ra $\lambda=4$, do đó $x=2$, $y=1$ và giá trị nhỏ nhất của mục tiêu là $2^2+2\times1^2=6$.

Chạy lại bằng `scipy.optimize.minimize` với ràng buộc đẳng thức `x+y=3` cho cùng kết quả $x=2,\ y=1$, mục tiêu $\approx6$, khớp với lời giải tay. Cơ chế vừa dùng — đưa ràng buộc vào hàm mục tiêu qua một nhân tử rồi cho đạo hàm bằng 0 — chính là cách phần dưới sẽ lặp lại với SVM, chỉ khác là có nhiều biến hơn và các ràng buộc là bất đẳng thức nên nhân tử ($\alpha_i,\mu_i$) phải không âm.

*Ý tưởng minh họa tham khảo từ Phạm Đình Khanh, Machine Learning Algorithms to Practice, ch. 7 SVM: [phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html](https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html).*

## 1. Nhắc lại Primal

$$
\begin{aligned}
\min_{w,b,\xi}\quad &\frac12\|w\|^2+C\sum_i\xi_i\\
\text{s.t.}\quad &1-y_i(w^Tx_i+b)-\xi_i\leq0,\\
&-\xi_i\leq0.
\end{aligned}\qquad\text{(3.1)}
$$

Ta đưa mỗi ràng buộc bất đẳng thức vào Lagrangian bằng một biến nhân tử không âm:

- $\alpha_i \geq 0$ cho ràng buộc margin.
- $\mu_i \geq 0$ cho ràng buộc $\xi_i \geq 0$.

## 2. Lagrangian

$$
\begin{aligned}
L(w,b,\xi,\alpha,\mu)
=&\frac12\|w\|^2+C\sum_i\xi_i\\
&-\sum_i\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]\\
&-\sum_i\mu_i\xi_i.
\end{aligned}\qquad\text{(3.2)}
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
\boxed{w=\sum_i\alpha_i y_i x_i}\qquad\text{(3.3)}
$$

Ý nghĩa: vector pháp tuyến tối ưu là tổ hợp có trọng số của các mẫu dữ liệu. Mẫu có alpha bằng 0 không đóng góp vào w.

### Theo b

$$
\frac{\partial L}{\partial b}=-\sum_i\alpha_i y_i=0
$$

Suy ra:

$$
\boxed{\sum_i\alpha_i y_i=0}\qquad\text{(3.4)}
$$

Đây là ràng buộc đẳng thức của bài toán Dual.

### Theo $\xi_i$

$$
\frac{\partial L}{\partial\xi_i}=C-\alpha_i-\mu_i=0
$$

Vì $\mu_i \geq 0$ và $\alpha_i \geq 0$:

$$
\mu_i=C-\alpha_i\quad\Longrightarrow\quad 0\leq\alpha_i\leq C\qquad\text{(3.5)}
$$

Đây là nguồn gốc của cận trên C trong bài toán đối ngẫu, không phải một điều kiện được đặt tùy ý.

## 4. Thay trở lại Lagrangian

Sau khi loại $w,b,\xi,\mu$, ta nhận được:

$$
\boxed{
\begin{aligned}
\max_{\alpha}\quad &\sum_i\alpha_i-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i^Tx_j)\\
\text{s.t.}\quad &\sum_i\alpha_i y_i=0,\\
&0\leq\alpha_i\leq C.
\end{aligned}}\qquad\text{(3.6)}
$$

Đây là bài toán đối ngẫu bậc hai. Ta tối đa hóa một hàm lõm theo alpha (tương đương tối thiểu hóa hàm lồi sau khi đổi dấu), với các ràng buộc tuyến tính.

Sơ đồ sau tóm tắt toàn bộ quá trình suy diễn ở trên (Primal → Lagrangian → đạo hàm → Dual); đọc từ trên xuống để nối lại các bước (3.1)–(3.6); nhánh chấm cuối cùng là bước thay tích vô hướng bằng kernel, trình bày ở doc 06.

```mermaid
flowchart TB
  subgraph PRIMAL["1 · Primal"]
    direction TB
    A["min ½‖w‖²+CΣξᵢ<br/>r.b. margin & ξᵢ≥0"]
  end
  subgraph LAGRANGE["2 · Lagrangian"]
    direction TB
    B["L(w,b,ξ,α,μ)<br/>αᵢ≥0, μᵢ≥0"]
  end
  subgraph DAOHAM["3 · Đạo hàm = 0"]
    direction LR
    C["∂L/∂w=0<br/>w=Σαᵢyᵢxᵢ"] --> D["∂L/∂b=0<br/>Σαᵢyᵢ=0"] --> E["∂L/∂ξᵢ=0<br/>C−αᵢ−μᵢ=0"]
  end
  subgraph DUAL["4 · Dual"]
    direction TB
    F["max Σαᵢ−½ΣΣαᵢαⱼyᵢyⱼxᵢᵀxⱼ<br/>r.b. Σαᵢyᵢ=0 · 0≤αᵢ≤C"]
  end
  PRIMAL --> LAGRANGE --> DAOHAM --> DUAL
  DUAL -. "thay xᵢᵀxⱼ bằng K(xᵢ,xⱼ)" .-> KERNEL["Kernel Trick"]
  classDef standard fill:#f4f6f8,stroke:#84929d,color:#1e2933,stroke-width:1.3px
  classDef focus fill:#e7efef,stroke:#6e8a87,color:#1e2933,stroke-width:1.4px
  class A,B,C,D,E standard
  class F,KERNEL focus
```

Nguồn: [`diagrams/lagrange_dual.mmd`](diagrams/lagrange_dual.mmd).

## 5. Primal và Dual khác nhau thế nào?

| Primal | Dual |
|---|---|
| Biến chính $w,b,\xi$ | Biến chính $\alpha$ |
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

`K` là Gram matrix, với $K_{ij}=x_i^T x_j$ ở Linear Kernel. Khi dùng kernel, phép tích vô hướng được thay bằng $K(x_i,x_j)$.

## 8. Bạn cần tự trả lời

1. $\alpha_i$ xuất hiện từ ràng buộc nào?
2. Vì sao $\alpha_i \leq C$?
3. Ràng buộc $\sum_i\alpha_i y_i=0$ xuất hiện từ đạo hàm theo biến nào?
4. Vì sao Dual thuận tiện để đưa Kernel vào?
