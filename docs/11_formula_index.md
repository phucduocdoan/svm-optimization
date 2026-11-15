# 11 — Chỉ mục công thức

Các công thức chính được đánh số để có thể dẫn chiếu thống nhất giữa Markdown và báo cáo Word. GitHub hỗ trợ MathJax cho block `$$...$$`; `\tag{...}` đặt số ngay trong công thức.

## (1.1) Hàm quyết định và siêu phẳng

$$
f(x)=w^T x+b=0 \tag{1.1}
$$

## (1.2) Khoảng cách và độ rộng margin

$$
d(x,H)=\frac{|w^T x+b|}{\|w\|_2},\qquad \mathrm{margin}=\frac{2}{\|w\|_2} \tag{1.2}
$$

## (2.1) Primal Soft-Margin

$$
\min_{w,b,\xi}\;\frac12\|w\|^2+C\sum_{i=1}^{n}\xi_i
\quad\mathrm{s.t.}\quad y_i(w^T x_i+b)\ge 1-\xi_i,\;\xi_i\ge0 \tag{2.1}
$$

## (2.2) Hinge loss

$$
\xi_i=\max\left(0,1-y_i f(x_i)\right) \tag{2.2}
$$

## (3.1) Lagrangian và stationarity

$$
L=\frac12\|w\|^2+C\sum_i\xi_i-\sum_i\alpha_i[y_i(w^T x_i+b)-1+\xi_i]-\sum_i\mu_i\xi_i \tag{3.1}
$$

$$
w=\sum_i\alpha_i y_i x_i,\qquad \sum_i\alpha_i y_i=0,\qquad 0\le\alpha_i\le C \tag{3.2}
$$

## (3.3) Dual SVM

$$
\max_{\alpha}\;\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j) \tag{3.3}
$$

## (4.1) Ràng buộc SMO

$$
y_i\Delta\alpha_i+y_j\Delta\alpha_j=0 \tag{4.1}
$$

## (5.1) Kernel Linear và RBF

$$
K_{\mathrm{linear}}(x,z)=x^Tz,\qquad K_{\mathrm{RBF}}(x,z)=\exp\left(-\gamma\|x-z\|^2\right) \tag{5.1}
$$

## (5.2) Hàm quyết định có kernel

$$
f(x)=\sum_i\alpha_i y_i K(x_i,x)+b \tag{5.2}
$$

## (6.1) Metrics chính

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad \mathrm{Recall}=\frac{TP}{TP+FN},\qquad F1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}} \tag{6.1}
$$

Xem các công thức theo ngữ cảnh tại [01 — Hình học SVM](01_svm_geometry.md), [02 — Primal Soft-Margin](02_primal_soft_margin.md), [03 — Lagrangian và Dual](03_lagrange_dual.md), [05 — SMO](05_smo_walkthrough.md) và [06 — Kernel Trick](06_kernel.md).
