# 09 — Hướng dẫn đọc code theo từng module

Tài liệu này là bản đồ từ lý thuyết sang từng file Python. Đừng đọc project từ trên xuống dưới; hãy đọc theo luồng dữ liệu và toán học.

## 1. Sơ đồ luồng gọi hàm

```mermaid
flowchart LR
  E["experiments/<br/>chạy thí nghiệm"] --> DATA["data.py<br/>tải và mã hóa dữ liệu"]
  E --> MODEL["models/svm.py<br/>KernelSVM"]
  MODEL --> K["models/kernel.py<br/>Linear · RBF"]
  MODEL --> SMO["optimization/smo.py<br/>SMOOptimizer"]
  SMO --> DUAL["math/dual.py<br/>hàm mục tiêu đối ngẫu"]
  E --> EVAL["evaluation.py<br/>metrics phân loại"]
  E --> VIZ["visualization.py<br/>hàm vẽ kết quả"]
  E -. "phân tích Primal" .-> PRIMAL["math/primal.py"]
  classDef standard fill:#f4f6f8,stroke:#84929d,color:#1e2933,stroke-width:1.3px
  classDef focus fill:#e7efef,stroke:#6e8a87,color:#1e2933,stroke-width:1.4px
  class E,DATA,MODEL,K,SMO,DUAL,EVAL,VIZ,PRIMAL standard
```

Nguồn: [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

## 2. Nhóm toán học: `src/svm_optimization/math/`

### `geometry.py`

**Nhiệm vụ:** công thức hình học độc lập với thuật toán huấn luyện.

- `decision_score(X,w,b)`: nhân ma trận `X @ w` rồi cộng bias.
- `predict_linear`: lấy dấu score.
- `vector_norm`: gọi chuẩn Euclid.
- `margin_width`: tính $2/\|w\|$.
- `point_to_hyperplane_distance`: tính khoảng cách có dấu tuyệt đối.

Đây là module nên đọc đầu tiên vì không có vòng lặp tối ưu phức tạp.

### `primal.py`

**Nhiệm vụ:** tính hàm mục tiêu và kiểm tra ràng buộc Primal.

- `hinge_losses`: mỗi phần tử là `max(0, 1-y_i*score_i)`.
- `soft_margin_objective`: cộng regularization và tổng hinge loss.
- `primal_constraints`: đo mức vi phạm margin/slack.

Thử thay score trong ví dụ `docs/02_primal_soft_margin.md` và quan sát loss tăng giảm.

### `dual.py`

**Nhiệm vụ:** làm việc trực tiếp với alpha.

- `dual_objective`: tính `sum(alpha) - 0.5 * (alpha*y)^T K (alpha*y)`.
- `dual_gradient`: gradient của Dual objective.
- `check_dual_feasibility`: kiểm tra cận alpha và equality constraint.

Hãy phân biệt `K` là ma trận kernel giữa mọi cặp mẫu, không phải ma trận nhãn.

### `kkt.py`

**Nhiệm vụ:** kiểm tra residual KKT và phân loại alpha.

- `kkt_report`: với Linear SVM, tính các residual từ `X,y,w,b,xi,alpha,C`.
- `alpha_roles`: phân nhóm alpha gần 0, ở giữa, gần C.

`residual` là độ lệch khỏi điều kiện; càng gần 0 càng tốt. Không phải mọi giá trị alpha đều phải đúng chính xác 0, C hoặc một số cố định do sai số tính toán.

## 3. Nhóm kernel và model: `src/svm_optimization/models/`

### `kernel.py`

**Nhiệm vụ:** trả về Gram matrix.

- Linear: `X @ Y.T`.
- RBF: khoảng cách bình phương từng cặp, rồi `exp(-gamma*distance²)`.

Ví dụ kernel trick trên hai tập mẫu `X` và `Z`:

```python
from svm_optimization.models.kernel import linear_kernel, rbf_kernel

K_linear = linear_kernel(X, Z)
K_rbf = rbf_kernel(X, Z, gamma=0.5)
```

Mỗi phần tử `K[i, j]` so sánh mẫu thứ `i` của `X` với mẫu thứ `j` của `Z`. Linear trả tích vô hướng; RBF trả `exp(-gamma * ||X[i] - Z[j]||²)`. Ví dụ giả định `X` và `Z` đã được chuẩn bị và có cùng số cột.

### `svm.py`

**Nhiệm vụ:** API mô hình mà người dùng gọi.

Luồng `fit(X,y)`:

1. Kiểm tra shape, NaN và nhãn `-1/+1`.
2. Tính gamma nếu dùng RBF.
3. Tạo Gram matrix train-train.
4. Gọi `SMOOptimizer.fit(K,y)`.
5. Lưu alpha, support vectors và bias.
6. Với Linear Kernel, tính $w=\sum_i\alpha_i y_i x_i$.

Luồng `decision_function(X)`:

1. Tính kernel giữa dữ liệu mới và dữ liệu train.
2. Nhân kernel với `alpha*y`.
3. Cộng bias.

Luồng `predict(X)` chỉ lấy dấu của decision score.

## 4. Nhóm tối ưu: `src/svm_optimization/optimization/smo.py`

Đây là module quan trọng và khó nhất. Nên đọc bằng cách đối chiếu từng khối với `docs/05_smo_walkthrough.md`.

| Khối code | Câu hỏi cần trả lời |
|---|---|
| Khởi tạo alpha/b/errors | Vì sao ban đầu error bằng `-y`? |
| `violates_kkt` | Điều kiện nào khiến mẫu cần update? |
| Tính L/H | Cận nào giữ alpha nằm trong [0,C]? |
| $\eta$ và `aj_new` | Đây là tối ưu một chiều theo $\alpha_j$ như thế nào? |
| `ai_new` | Làm sao giữ $\sum_i\alpha_i y_i=0$? |
| `b1`, `b2` | Vì sao hai điểm cập nhật có thể đưa ra hai bias? |
| `errors += ...` | Vì sao có thể cập nhật cache thay vì tính lại mọi score? |
| Điều kiện dừng | Thuật toán dừng do không còn update hay do đạt max_iter? |

## 5. Dataset: `data.py`

`load_banknote()` tải UCI qua `ucimlrepo`, kiểm tra missing values, chuyển target thành -1/+1 và cache thành CSV.

Không dùng thông tin từ test set để chuẩn hóa. Việc chia dữ liệu và chuẩn hóa thuộc experiment script, không nhét vào hàm tải dữ liệu để có thể kiểm soát split rõ ràng.

## 6. Evaluation và Visualization

### `evaluation.py`

`evaluate_binary` trả dictionary gồm Accuracy, Precision, Recall, F1, confusion matrix và classification report. Positive class được đặt là +1.

### `visualization.py`

- `plot_2d_boundary`: chỉ dùng khi model đã train bằng đúng hai feature.
- `plot_confusion_matrix`: trực quan hóa nhầm lẫn.
- `plot_objective_history`: xem dual objective sau từng pass.

Không dùng plot 2D như bằng chứng thay thế kết quả trên đủ bốn feature. Đây chỉ là biểu diễn trực quan.

## 7. Đọc experiment theo mục tiêu

- `01_baseline_sklearn.py`: cách dùng chuẩn `Pipeline(StandardScaler(), SVC(...))`.
- `02_hand_example.py`: ví dụ nhỏ để kiểm tra công thức và SMO.
- `03_custom_smo.py`: luồng dữ liệu thực tế đi qua model tự cài.
- `04_compare_C_kernel.py`: khảo sát nhạy cảm theo C/kernel.
- `05_visualize_2d.py`: mô hình riêng trên hai feature đầu để vẽ ranh giới trực quan.

## 8. Một vòng SMO

```mermaid
flowchart LR
  subgraph CHON["1 · Chọn cặp"]
    direction TB
    A["Duyệt i vi phạm KKT<br/>chọn ứng viên j"] --> B["Tính L, H và η<br/>xét cặp hợp lệ"]
  end
  subgraph CAPNHAT["2 · Cập nhật"]
    direction TB
    C["Cập nhật αⱼ và αᵢ<br/>giữ Σ αᵢyᵢ = 0"] --> D["Cập nhật b<br/>và bộ nhớ sai số"]
  end
  subgraph DUNG["3 · Lặp và dừng"]
    direction TB
    E["Lặp và kiểm tra<br/>điều kiện dừng"] --> F["Dừng: max_passes lượt<br/>không đổi hoặc max_iter"]
  end
  CHON --> CAPNHAT --> DUNG
  classDef standard fill:#f4f6f8,stroke:#84929d,color:#1e2933,stroke-width:1.3px
  classDef focus fill:#e7efef,stroke:#6e8a87,color:#1e2933,stroke-width:1.4px
  class A,B,C,D,E standard
  class F focus
```

Nguồn: [`diagrams/smo_pair_update.mmd`](diagrams/smo_pair_update.mmd).

## 9. Bài tập code nhỏ

1. Sửa `C=1` thành `C=0.1` trong ví dụ 4 điểm. Alpha thay đổi thế nào?
2. Thêm một điểm gần biên vào ví dụ nhỏ và xem support vectors thay đổi.
3. Thử Linear và RBF trên XOR trong `tests/test_smo.py`.
4. Thêm một dòng print `np.dot(alpha, y)` sau fit để kiểm tra equality constraint.
5. Thêm test để xác nhận tất cả alpha nằm trong `[0,C]`.
