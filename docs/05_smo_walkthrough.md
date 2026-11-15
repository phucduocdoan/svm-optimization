# 05 — SMO: cách giải bài toán đối ngẫu

## 1. Bài toán cần giải

SMO giải bài toán Dual:

$$
\max_\alpha W(\alpha)=\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j) \tag{5.1}
$$

với:

$$
\sum_i\alpha_i y_i=0,\qquad 0\leq\alpha_i\leq C \tag{5.2}
$$

Nếu tối ưu từng `alpha_i` độc lập, ta có thể làm hỏng ràng buộc đẳng thức. SMO xử lý bằng cách cập nhật **hai alpha cùng lúc** để vẫn giữ:

![Trực giác đi xuống trên hàm mục tiêu](https://commons.wikimedia.org/wiki/Special:FilePath/Gradient_descent.png?width=1000)

Đây là hình minh họa trực giác tối ưu hóa nói chung. SMO không đi theo một gradient descent đơn giản; nó tối ưu một bài toán con hai biến và giữ ràng buộc `sum(alpha*y)=0`. Nguồn: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent.png), public domain.

$$
y_i\Delta\alpha_i+y_j\Delta\alpha_j=0 \tag{5.3}
$$

## 2. Các ký hiệu cần hiểu

- `alpha_i`, `alpha_j`: hai hệ số Lagrange đang cập nhật.
- `E_i = f(x_i)-y_i`: sai số dự đoán trên mẫu i.
- `K_ij = K(x_i,x_j)`: giá trị kernel.
- `L,H`: cận dưới và trên bảo đảm hai alpha sau cập nhật vẫn khả thi.
- `b`: bias.

## 3. Kiểm tra KKT để quyết định cập nhật

Đặt `r_i = y_i E_i`. Mẫu i có thể vi phạm KKT nếu:

$$
(r_i < -tol\;\text{và}\;\alpha_i<C)
\quad\text{hoặc}\quad
(r_i > tol\;\text{và}\;\alpha_i>0).
$$

Nếu không vi phạm đáng kể, tạm thời không cần cập nhật mẫu đó.

## 4. Tính khoảng [L,H]

Nếu `y_i != y_j`:

$$
L=\max(0,\alpha_j-\alpha_i),\quad
H=\min(C,C+\alpha_j-\alpha_i).
$$

Nếu `y_i = y_j`:

$$
L=\max(0,\alpha_i+\alpha_j-C),\quad
H=\min(C,\alpha_i+\alpha_j).
$$

Hai công thức khác nhau vì dấu của hai nhãn quyết định quan hệ giữa alpha_i và alpha_j khi giữ equality constraint.

## 5. Cập nhật alpha_j

Tính:

$$
\eta=2K_{ij}-K_{ii}-K_{jj} \tag{5.4}
$$

Với kernel hợp lệ, Gram matrix là positive semidefinite nên `eta` thường không dương. Khi `eta` âm đủ xa 0:

$$
\alpha_j^{new}=\alpha_j^{old}-\frac{y_j(E_i-E_j)}{\eta} \tag{5.5}
$$

Sau đó clip về đoạn [L,H]:

$$
\alpha_j^{new}\leftarrow\min(H,\max(L,\alpha_j^{new})).
$$

## 6. Cập nhật alpha_i

Giữ ràng buộc đẳng thức bằng:

$$
\alpha_i^{new}=\alpha_i^{old}+y_i y_j(\alpha_j^{old}-\alpha_j^{new}) \tag{5.6}
$$

Như vậy hai alpha thay đổi có liên hệ, không phải hai cập nhật tùy ý.

## 7. Cập nhật bias b

Tính hai ứng viên:

$$
\begin{aligned}
b_1=&b-E_i-y_i(\alpha_i^{new}-\alpha_i^{old})K_{ii}\\
&-y_j(\alpha_j^{new}-\alpha_j^{old})K_{ij},
\end{aligned}
$$

$$
\begin{aligned}
b_2=&b-E_j-y_i(\alpha_i^{new}-\alpha_i^{old})K_{ij}\\
&-y_j(\alpha_j^{new}-\alpha_j^{old})K_{jj}.
\end{aligned}
$$

- Nếu `0<alpha_i_new<C`, chọn `b=b1`.
- Nếu không, nhưng `0<alpha_j_new<C`, chọn `b=b2`.
- Nếu cả hai ở biên, lấy trung bình `(b1+b2)/2`.

## 8. Tại sao SMO là thuật toán tối ưu?

SMO không tìm w bằng gradient descent thông thường. Nó tối ưu trực tiếp hàm mục tiêu Dual theo từng cặp alpha, đồng thời giữ ràng buộc tuyến tính. Việc giải bài toán con hai biến có thể thực hiện bằng công thức giải tích nêu trên.

Một lượt quét xem xét các mẫu; nếu còn vi phạm KKT thì thuật toán thử chọn cặp khác và cập nhật. Trong bản học tập này, dừng khi đạt số lượt liên tiếp không cập nhật hoặc vượt `max_iter`.

## 9. Liên hệ với code

File `src/svm_optimization/optimization/smo.py`.

Bạn nên đọc theo thứ tự:

1. `fit(K,y)`: kiểm tra đầu vào, khởi tạo `alpha=0`, `b=0`.
2. `errors=-y`: vì ban đầu `f(x_i)=0`, nên `E_i=0-y_i`.
3. `violates_kkt`: quyết định có cần cập nhật i.
4. Tính `L,H`, `eta`, alpha mới.
5. Tính `b1,b2` và chọn bias.
6. Cập nhật cache sai số để không phải tính lại toàn bộ hàm dự đoán mỗi lần.
7. Lưu objective sau mỗi pass.

**Giới hạn bản cài đặt:** cặp có `eta` gần 0 bị bỏ qua để giữ code dễ học. Đây là một lựa chọn đơn giản, không phải cách xử lý đầy đủ mọi trường hợp suy biến. Các thư viện tối ưu SVM thực tế như LIBSVM có nhiều chiến lược chọn working set và xử lý biên tinh vi hơn.

## 10. Bạn cần tự trả lời

1. Vì sao SMO cập nhật hai alpha chứ không phải một?
2. Việc clip alpha_j vào [L,H] đảm bảo điều gì?
3. `E_i` khác margin `y_i*f(x_i)` như thế nào?
4. Vì sao bias có hai ứng viên b1 và b2?
5. Nếu `eta` gần 0, vì sao công thức cập nhật có thể không ổn định?
