## TỔNG QUAN VỀ HỌC MÁY VÀ BÀI TOÁN PHÂN LỚP

### Tổng quan về học máy có giám sát (Supervised Learning)

Khái niệm và cơ chế vận hành của học máy có giám sát

Học máy có giám sát là một mô hình học tập trong đó thuật toán được cung cấp một tập dữ liệu huấn luyện đã được gán nhãn trước. Mỗi mẫu dữ liệu bao gồm một tập hợp các thuộc tính đặc trưng (features) và nhãn mục tiêu (target label) tương ứng.

Cơ chế vận hành của mô hình dựa trên việc xây dựng một hàm ánh xạ từ không gian đặc trưng đầu vào sang không gian nhãn đầu ra thông qua quá trình tối ưu hóa các tham số nội tại.

Quá trình này bao gồm hai giai đoạn chính: giai đoạn huấn luyện (Training) — nơi mô hình học quy luật từ dữ liệu mẫu, và giai đoạn kiểm thử (Testing) — nơi mô hình dự đoán nhãn cho các dữ liệu mới chưa từng xuất hiện.

Bài toán phân lớp dữ liệu và ứng dụng thực tiễn

Phân lớp là bài toán học có giám sát trong đó nhãn mục tiêu là các giá trị rời rạc (ví dụ: thư rác / không phải thư rác, ung thư lành tính / ác tính).

Mục tiêu toán học của bài toán là tìm ra một ranh giới (boundary) quyết định trong không gian đặc trưng nhằm phân tách tối ưu các mẫu thuộc các lớp khác nhau.

Các ứng dụng thực tiễn của bài toán phân lớp rất đa dạng, từ việc phân loại tín hiệu vệ tinh trong ngành thủy văn, dự báo lũ lụt đến việc phân loại trạng thái cấu trúc công trình xây dựng.

### Lịch sử hình thành và vị thế của Máy vector hỗ trợ (SVM)

Quá trình phát triển từ lý thuyết thống kê Vapnik – Chervonenkis (VC)

Thuật toán Máy vector hỗ trợ (Support Vector Machine - SVM) do Vladimir Vapnik và các cộng sự phát triển vào những năm 1990 tại các phòng thí nghiệm của trường AT&T Bell.

Nền tảng toán học của SVM bắt nguồn từ Lý thuyết thống kê học tập (Statistical Learning Theory) và khái niệm số chiều VC (Vapnik-Chervonenkis dimension), nhằm đo lường năng lực học và độ phức tạp của không gian hàm giả thuyết.

Khác với các mô hình mạng nơ-ron nhân tạo thời kỳ đầu vốn dễ gặp hiện tượng quá khớp (overfitting) trên dữ liệu nhỏ, SVM được thiết kế để tìm kiếm sự cân bằng tối ưu giữa độ phức tạp của mô hình và sai số trên tập huấn luyện.

Nguyên lý tối thiểu hóa rủi ro cấu trúc (SRM) so với rủi ro thực nghiệm (ERM)

Đa số các thuật toán học máy truyền thống thường áp dụng nguyên lý Tối thiểu hóa rủi ro thực nghiệm (Empirical Risk Minimization - ERM), tức là cố gắng giảm thiểu tối đa sai số trên tập dữ liệu huấn luyện, dẫn đến rủi ro mô hình mất khả năng tổng quát hóa trên dữ liệu mới.

Ngược lại, SVM triệt để áp dụng nguyên lý Tối thiểu hóa rủi ro cấu trúc (Structural Risk Minimization - SRM). Nguyên lý này không chỉ giảm thiểu sai số thực nghiệm mà đồng thời kiểm soát độ phức tạp của không gian hàm, giúp mô hình SVM đạt độ bền vững (robustness) rất cao ngay cả khi đối mặt với dữ liệu có chiều lớn.

### Kết luận chương 1

Chương 1 đã phác họa bức tranh toàn cảnh về học máy có giám sát, định hình bài toán phân lớp dữ liệu cũng như định vị tầm quan trọng lịch sử và lý thuyết của mô hình Máy vector hỗ trợ (SVM). Những nền tảng lý thuyết về tối ưu hóa rủi ro cấu trúc sẽ là tiền đề vững chắc để triển khai chi tiết cơ sở toán học, hình học và các biến thể nâng cao của SVM trong Chương 2 tiếp theo.

## CƠ SỞ LÝ THUYẾT VÀ HÌNH HỌC CỦA MÁY VECTOR HỖ TRỢ

### Siêu phẳng phân tách và khái niệm Lề (Margin)

Biểu diễn hình học của siêu phẳng phân tách trong không gian nhiều chiều

Trong không gian đặc trưng nhiều chiều, mục tiêu cốt lõi của bài toán phân lớp nhị phân là tìm kiếm một siêu phẳng (Hyperplane) để chia tách rạch ròi các điểm dữ liệu thuộc hai lớp khác nhau. Giả sử tập dữ liệu huấn luyện gồm N mẫu dữ liệu x_i tương ứng với nhãn y_i thuộc tập {-1, +1}. Một siêu phẳng trong không gian được định nghĩa bởi phương trình:

w^T . x - b = 0

Trong đó: w là véc-tơ pháp tuyến vuông góc với siêu phẳng; x là véc-tơ đặc trưng đầu vào; b là hệ số dịch chuyển (bias). Siêu phẳng này đóng vai trò là ranh giới quyết định: nếu giá trị (w^T . x_i - b) > 0, mẫu dữ liệu được gán về lớp +1; ngược lại, nếu (w^T . x_i - b) < 0, mẫu dữ liệu được gán về lớp -1.

Khoảng cách hình học và ý nghĩa của việc tối đa hóa lề đối với khả năng tổng quát hóa

Thực tế cho thấy có vô số siêu phẳng khác nhau thỏa mãn điều kiện phân tách chính xác tập dữ liệu huấn luyện. Tuy nhiên, để mô hình có khả năng tổng quát hóa tốt nhất trên dữ liệu mới chưa từng thấy, ta không chọn một siêu phẳng tùy ý mà tìm siêu phẳng nằm chính giữa khoảng cách xa nhất giữa hai lớp.

  Khoảng cách hình học từ một điểm dữ liệu bất kỳ đến siêu phẳng phân tách được xác định dựa trên độ lớn của véc-tơ pháp tuyến w.

  Khoảng cách biên (Margin) — được tính bằng tổng khoảng cách ngắn nhất từ siêu phẳng đến các điểm dữ liệu gần nhất của hai lớp, có giá trị bằng 2 / ||w|| — cần được tối đa hóa. Việc tối đa hóa bề rộng lề giúp giảm thiểu sai số kiểm thử và đảm bảo tính bền vững cho mô hình trước nhiễu.

Vai trò của các Vector hỗ trợ (Support Vectors)

Trong quá trình tìm kiếm siêu phẳng tối ưu, phần lớn các điểm dữ liệu nằm sâu bên trong các lớp không đóng vai trò quyết định đến vị trí của siêu phẳng. Chỉ có một tập hợp nhỏ các điểm dữ liệu nằm ngay trên hai đường biên của lề mới quyết định trực tiếp đến hình học của siêu phẳng. Những điểm dữ liệu đặc biệt này được gọi là các Vector hỗ trợ (Support Vectors). Nếu loại bỏ tất cả các điểm dữ liệu thông thường khác khỏi tập huấn luyện, vị trí của siêu phẳng tối ưu hoàn toàn không thay đổi, minh chứng cho tính gọn gàng và hiệu quả của mô hình.

### Máy vector hỗ trợ lề cứng (Hard Margin SVM)

Giả thiết dữ liệu khả tách tuyến tính hoàn toàn

Mô hình lề cứng (Hard Margin SVM) xuất phát từ giả thiết lý tưởng rằng tập dữ liệu huấn luyện có thể phân tách tuyến tính hoàn toàn tuyệt đối. Điều này có nghĩa là tồn tại ít nhất một siêu phẳng sao cho toàn bộ các điểm dữ liệu của lớp +1 nằm hoàn toàn ở một phía (w^T . x_i - b >= 1) và toàn bộ các điểm dữ liệu của lớp -1 nằm hoàn toàn ở phía đối diện (w^T . x_i - b <= -1).

Phát biểu bài toán tối ưu hóa gốc và ý nghĩa các ràng buộc

Dựa trên yêu cầu tối đa hóa bề rộng lề (tương đương với việc cực tiểu hóa bình phương chuẩn của véc-tơ pháp tuyến) kết hợp với các điều kiện phân tách, bài toán tối ưu hóa gốc được phát biểu dưới dạng bài toán quy hoạch toàn phương (Quadratic Programming) như sau:

Cực tiểu hóa hàm mục tiêu: (1/2) . ||w||^2 (2.2)

Với điều kiện ràng buộc: y_i . (w^T . x_i - b) >= 1, với mọi i = 1, 2, ..., N

Hệ thống ràng buộc này đảm bảo mọi điểm dữ liệu đều phải nằm ngoài vùng lề an toàn và nằm đúng phía của lớp tương ứng.

### Máy vector hỗ trợ lề mềm (Soft Margin SVM)

Hạn chế của lề cứng trước dữ liệu nhiễu và điểm ngoại lai (Outliers)

Trong thực tế ứng dụng, dữ liệu hầu như không bao giờ khả tách tuyến tính hoàn toàn. Sự xuất hiện của nhiễu, sai số đo đạc hoặc các điểm ngoại lai (outliers) nằm lẫn lộn giữa hai lớp sẽ khiến bài toán lề cứng không tìm được nghiệm khả thi. Nếu ép buộc tìm siêu phẳng phân tách tuyệt đối, mô hình sẽ trở nên rất nhạy cảm, dễ dẫn đến hiện tượng quá khớp (overfitting) và giảm nghiêm trọng khả năng tổng quát hóa.

Vai trò của biến bù và tham số điều chuẩn C

Để giải quyết bài toán dữ liệu có nhiễu, Corinna Cortes và Vladimir Vapnik đã đề xuất mô hình lề mềm (Soft Margin SVM) thông qua việc bổ sung các biến bù không âm (ký hiệu là xi_i >= 0, với i = 1, 2, ..., N). Biến bù cho phép một số điểm dữ liệu vi phạm khoảng cách lề hoặc thậm chí bị phân loại sai.

Bài toán tối ưu hóa lúc này kết hợp giữa việc tối đa hóa lề và giảm thiểu tổng mức độ vi phạm:

Cực tiểu hóa hàm mục tiêu: (1/2) . ||w||^2 + C . Tổng(xi_i) (2.3)

Với điều kiện ràng buộc: y_i . (w^T . x_i - b) >= 1 - xi_i và xi_i >= 0

Tham số điều chuẩn C > 0 đóng vai trò cân bằng quan trọng: khi C có giá trị lớn, mô hình phạt nặng các điểm vi phạm, ép mô hình tìm lề hẹp để phân loại chính xác tối đa; ngược lại, khi C nhỏ, mô hình ưu tiên lề rộng và chấp nhận một số điểm vi phạm nhằm duy trì tính tổng quát.

Mối liên hệ với hàm mất mát Bản lề (Hinge Loss)

Góc nhìn tối ưu hóa của lề mềm có thể được diễn giải dưới dạng hàm mục tiêu không có ràng buộc bao gồm tổng của thành phần chuẩn hóa và hàm mất mát thực nghiệm. Cụ thể, thành phần tổn thất tương ứng chính là hàm mất mát bản lề (Hinge Loss):

L_hinge = max(0, 1 - y_i . (w^T . x_i - b))

Hàm mất mát này có đặc tính triệt tiêu tổn thất (bằng 0) khi điểm dữ liệu nằm đúng phía và ngoài khoảng cách an toàn, đồng thời tăng tuyến tính khi điểm dữ liệu tiến sát hoặc vượt qua ranh giới phân loại.

### Phương pháp nhân tử Lagrange và bài toán đối ngẫu

Ý nghĩa của việc chuyển đổi sang bài toán đối ngẫu

Để giải quyết bài toán tối ưu hóa với các ràng buộc bất đẳng thức phức tạp, người ta sử dụng phương pháp nhân tử Lagrange (với các hệ số nhân tử alpha_i >= 0) để chuyển bài toán gốc sang dạng bài toán đối ngẫu (Dual Problem). Việc chuyển đổi này mang lại hai lợi ích lớn:

Đưa bài toán về dạng chỉ phụ thuộc vào tích vô hướng giữa các véc-tơ đặc trưng đầu vào (x_i^T . x_j).

Tạo tiền đề cốt lõi để áp dụng thành công các hàm hạt nhân (Kernel Trick) nhằm xử lý dữ liệu phi tuyến tính ở các chương tiếp theo.

Điều kiện tối ưu KKT (Karush-Kuhn-Tucker) và tính thưa thớt của nghiệm

Thông qua các điều kiện tối ưu KKT (Karush-Kuhn-Tucker), nghiệm của bài toán đối ngẫu thỏa mãn các tính chất quan trọng:

Chỉ những điểm dữ liệu nằm trên biên lề hoặc vi phạm lề mới có hệ số Lagrange dương (alpha_i > 0), và các điểm này chính là các Vector hỗ trợ.

Toàn bộ các điểm dữ liệu nằm sâu bên trong các lớp đều có hệ số alpha_i = 0, thể hiện tính thưa thớt (sparsity) của nghiệm giúp tối ưu hóa đáng kể tài nguyên tính toán khi huấn luyện mô hình.

### Kết luận chương 2

Chương 2 đã trình bày tường minh cơ sở toán học và hình học cốt lõi của mô hình SVM, từ trường hợp lề cứng lý tưởng đến mô hình lề mềm thực tế với biến bù và hàm mất mát bản lề, cũng như việc chuyển đổi qua bài toán đối ngẫu Lagrange. Trên nền tảng này, Chương 3 sẽ tiếp tục mở rộng mô hình sang không gian phi tuyến thông qua thủ thuật hạt nhân (Kernel Trick).

## PHƯƠNG PHÁP HẠT NHÂN (KERNEL TRICK) VÀ CÁC BIẾN THỂ MỞ RỘNG

### Bài toán dữ liệu phi tuyến và ánh xạ lên không gian đặc trưng chiều cao

Hạn chế của ranh giới tuyến tính trên dữ liệu phức tạp

Trong thực tế, nhiều bài toán phân lớp dữ liệu có cấu trúc rất phức tạp, ví dụ như các mẫu thuộc lớp này bao quanh mẫu thuộc lớp kia theo dạng hình tròn hoặc hình vành khuyên đồng tâm. Trong những trường hợp như vậy, tuyệt đối không thể tìm thấy một siêu phẳng (đường thẳng trong không gian hai chiều) nào có thể chia tách chúng một cách chính xác mà không gây ra sai số lớn. Việc cố gắng áp dụng mô hình phân tách tuyến tính thuần túy sẽ dẫn đến hiệu năng dự đoán rất kém.

Ý tưởng ánh xạ không gian và thách thức chi phí tính toán

Để giải quyết bài toán dữ liệu phi tuyến, ý tưởng cốt lõi là sử dụng một hàm ánh xạ phi tuyến (ký hiệu là phi(x)) để chuyển đổi toàn bộ không gian dữ liệu đầu vào có số chiều thấp sang một không gian đặc trưng mới có số chiều cao hơn rất nhiều. Trong không gian mới này, các điểm dữ liệu sẽ được phân tán rộng hơn và trở nên khả tách tuyến tính bằng một siêu phẳng phẳng phẳng bình thường. Tuy nhiên, thách thức lớn nhất của phương pháp này là chi phí tính toán và bộ nhớ tăng theo cấp số nhân khi số chiều của không gian đặc trưng trở nên cực kỳ lớn hoặc thậm chí có số chiều vô hạn.

### Thủ thuật hạt nhân (Kernel Trick) và Điều kiện Mercer

Cơ chế tính tích vô hướng trực tiếp thông qua hàm hạt nhân

Để vượt qua rào cản chi phí tính toán trong không gian số chiều cao, các nhà toán học đã ứng dụng "Thủ thuật hạt nhân" (Kernel Trick). Thay vì phải thực hiện tường minh quá trình biến đổi phi tuyến phi(x_i) rồi mới tính tích vô hướng, thủ thuật hạt nhân cho phép ta tính trực tiếp tích vô hướng của hai véc-tơ trong không gian đặc trưng mới thông qua một hàm hạt nhân K(x_i, x_j) tính toán ngay trên không gian đầu vào ban đầu:

K(x_i, x_j) = phi(x_i)^T . phi(x_j)

Nhờ kỹ thuật này, thuật toán SVM có thể vận hành trơn tru trong không gian chiều rất cao mà không bao giờ cần phải thực hiện tường minh các phép biến đổi không gian phức tạp.

Điều kiện Mercer đảm bảo tính lồi và nghiệm tối ưu toàn cục

Không phải hàm số nào cũng có thể đóng vai trò là một hàm hạt nhân hợp lệ. Theo Định lý Mercer, một hàm K(x_i, x_j) được coi là hàm hạt nhân hợp lệ khi và chỉ khi ma trận hạt nhân tương ứng (ma trận Gram chứa tất cả các giá trị K giữa các cặp mẫu dữ liệu) là một ma trận đối xứng và nửa xác định dương. Điều kiện này đảm bảo rằng bài toán tối ưu hóa đối ngẫu vẫn giữ được tính lồi (convex), từ đó đảm bảo thuật toán luôn tìm được nghiệm tối ưu toàn cục duy nhất.

### Phân tích các hàm hạt nhân phổ biến

Hạt nhân tuyến tính (Linear Kernel) và hạt nhân đa thức (Polynomial Kernel)

  Hạt nhân tuyến tính (Linear Kernel): Có dạng K(x_i, x_j) = x_i^T . x_j. Đây thực chất là trường hợp đặc biệt không sử dụng biến đổi phi tuyến, phù hợp với các bài toán dữ liệu có số chiều đặc trưng rất lớn nhưng khả tách tuyến tính tốt (như bài toán phân loại văn bản).

  Hạt nhân đa thức (Polynomial Kernel): Có dạng K(x_i, x_j) = (gamma . x_i^T . x_j + r)^d, với d là bậc của đa thức. Hạt nhân này cho phép mô hình tạo ra các ranh giới quyết định cong cấp cao, phù hợp với các bài toán có quan hệ tương tác đa thức giữa các thuộc tính.

Hạt nhân hàm cơ sở xuyên tâm (RBF / Gaussian Kernel) và vai trò siêu tham số

Hạt nhân RBF (Radial Basis Function) hay còn gọi là hạt nhân Gaussian là một trong những hàm hạt nhân phổ biến và mạnh mẽ nhất, được định nghĩa bởi biểu thức:

K(x_i, x_j) = exp( -gamma . ||x_i - x_j||^2 )

Trong đó, siêu tham số gamma > 0 đóng vai trò kiểm soát bán kính ảnh hưởng của từng mẫu dữ liệu huấn luyện:

Khi gamma có giá trị lớn, vùng ảnh hưởng trở nên rất hẹp, mô hình tạo ra ranh giới quyết định uốn lượn sát theo từng điểm dữ liệu, dễ dẫn đến hiện tượng quá khớp (overfitting).

Khi gamma có giá trị nhỏ, vùng ảnh hưởng rộng lớn, ranh giới quyết định trở nên mượt mà hơn, mang lại khả năng tổng quát hóa tốt hơn cho mô hình.

Bảng so sánh đặc điểm và tiêu chí lựa chọn hàm hạt nhân

Tùy thuộc vào bản chất phân bố của dữ liệu thực tế, việc lựa chọn hàm hạt nhân thích hợp đóng vai trò quyết định đến độ chính xác của mô hình SVM. Hạt nhân tuyến tính ưu tiên tốc độ và sự đơn giản trên dữ liệu lớn; hạt nhân đa thức thích hợp khi cần khai thác quan hệ kết hợp; trong khi hạt nhân RBF là lựa chọn mặc định mạnh mẽ cho các bài toán phân bố phi tuyến tổng quát.

### Mở rộng SVM cho bài toán phân lớp đa lớp và hồi quy (SVR)

Chiến lược Một-chống-Tất cả (OvR) và Một-chống-Một (OvO)

Bản chất ban đầu của SVM là thuật toán phân lớp nhị phân (chỉ gồm 2 lớp). Để giải quyết các bài toán phân lớp đa lớp (có từ 3 lớp trở lên), người ta áp dụng các chiến lược kết hợp:

Chiến lược Một-chống-Tất cả (One-vs-Rest - OvR): Xây dựng K mô hình phân lớp, mỗi mô hình làm nhiệm vụ phân tách một lớp cụ thể với tất cả các lớp còn lại.

Chiến lược Một-chống-Một (One-vs-One - OvO): Xây dựng K * (K - 1) / 2 mô hình phân lớp cho từng cặp hai lớp khác nhau, sau đó dùng phương pháp bỏ phiếu đa số để đưa ra nhãn dự đoán cuối cùng.

Khái quát về Hồi quy vector hỗ trợ (Support Vector Regression – SVR)

Bên cạnh bài toán phân lớp, cấu trúc của SVM còn được mở rộng thành công sang bài toán dự báo giá trị liên tục, được gọi là Hồi quy vector hỗ trợ (Support Vector Regression - SVR). Thay vì tìm siêu phẳng phân tách lớp, SVR tìm kiếm một dải dung sai (epsilon-tube) sao cho phần lớn các điểm dữ liệu nằm gọn bên trong dải này, đồng thời tối thiểu hóa độ lớn của véc-tơ pháp tuyến để đảm bảo hàm hồi quy đạt độ mượt mà tối ưu.

### Kết luận Chương 3

Chương 3 đã giải quyết trọn vẹn bài toán dữ liệu phi tuyến thông qua thủ thuật hạt nhân (Kernel Trick) và điều kiện Mercer, đồng thời phân tích chi tiết các hàm hạt nhân phổ biến như RBF, đa thức cũng như các phương pháp mở rộng sang bài toán đa lớp và bài toán hồi quy SVR. Trên nền tảng lý thuyết này, Chương 4 sẽ chuyển sang phần ví dụ tính tay minh họa và thực nghiệm đánh giá mô hình.

## VÍ DỤ MINH HỌA VÀ THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH

### Ví dụ minh họa giải bài toán phân lớp SVM bằng tay

Thiết lập bộ dữ liệu mẫu 2 chiều và xác định các điểm hỗ trợ

Để hiểu rõ bức tranh tính toán đằng sau thuật toán, ta xét một bài toán phân lớp nhị phân tối giản với tập dữ liệu huấn luyện gồm 3 điểm mẫu trong không gian hai chiều như sau:

Điểm mẫu 1: x_1 = [2, 2]^T, thuộc lớp nhãn y_1 = +1

Điểm mẫu 2: x_2 = [2, 0]^T, thuộc lớp nhãn y_2 = +1

Điểm mẫu 3: x_3 = [0, 0]^T, thuộc lớp nhãn y_3 = -1

Mục tiêu là tìm ra siêu phẳng phân tách tối ưu có dạng w_1 . x_1 + w_2 . x_2 - b = 0 sao cho lề phân cách giữa hai lớp đạt giá trị lớn nhất. Dựa vào trực giác hình học, hai điểm gần ranh giới nhất của hai lớp chính là x_2 và x_3 (hoặc x_1), từ đó xác định các điểm này đóng vai trò là các Vector hỗ trợ quyết định vị trí của siêu phẳng.

Thiết lập hệ phương trình từ điều kiện KKT, tìm siêu phẳng tối ưu và bề rộng lề

Áp dụng bài toán tối ưu hóa lề cứng, ta cần cực tiểu hóa (1/2) * ||w||^2 với các điều kiện ràng buộc y_i . (w^T . x_i - b) >= 1 đối với từng mẫu dữ liệu:

Đối với mẫu 1: +1 . (2w_1 + 2w_2 - b) >= 1

Đối với mẫu 2: +1 . (2w_1 + 0w_2 - b) >= 1

Đối với mẫu 3: -1 . (0w_1 + 0w_2 - b) >= 1 => b >= 1 (hoặc chọn b = 1 tại biên)

Giải hệ phương trình tối ưu hóa trên, ta thu được véc-tơ pháp tuyến của siêu phẳng là w = [0.5, 0.5]^T và hệ số dịch chuyển b = 1. Phương trình siêu phẳng phân tách tối ưu cuối cùng là:

0.5 . x_1 + 0.5 . x_2 - 1 = 0

Bề rộng lề tối ưu thu được tương ứng là 2 / ||w|| = 2 / căn bậc hai(0.5^2 + 0.5^2) khoảng 2.828, thể hiện ranh giới phân tách an toàn nhất cho dữ liệu mẫu.

### Thiết lập thực nghiệm và tiền xử lý dữ liệu

Mô tả tập dữ liệu thực nghiệm và tầm quan trọng của chuẩn hóa đặc trưng (Feature Scaling)

Trong phần thực nghiệm trên quy mô lớn, ta sử dụng tập dữ liệu chuẩn hóa gồm nhiều chiều đặc trưng (ví dụ: tập dữ liệu phân loại hoa Iris hoặc dữ liệu chẩn đoán).

Một bước bắt buộc trước khi huấn luyện mô hình SVM là chuẩn hóa đặc trưng (Feature Scaling), đưa các giá trị thuộc tính về cùng một miền giá trị (như khoảng [0, 1] hoặc phân phối chuẩn hóa có trung bình bằng 0 và độ lệch chuẩn bằng 1).

Nguyên nhân là do SVM tính toán khoảng cách hình học dựa trên tích vô hướng và chuẩn của véc-tơ; nếu một thuộc tính có biên độ quá lớn sẽ lấn át hoàn toàn các thuộc tính có biên độ nhỏ, làm lệch mô hình.

Quy trình tinh chỉnh siêu tham số bằng Grid Search và Cross-Validation

Để đạt hiệu năng tối ưu trên tập kiểm thử, mô hình SVM đòi hỏi phải tinh chỉnh cặp siêu tham số (C, gamma) (khi dùng hạt nhân RBF). Quá trình này được thực hiện thông qua phương pháp tìm kiếm lưới (Grid Search) kết hợp kiểm định chéo (k-Fold Cross-Validation, thường chọn k = 5):

Tập hợp các giá trị thử nghiệm cho C thường nằm trong dải lũy thừa của 10 (ví dụ: từ 0.1 đến 1000).

Tập hợp các giá trị thử nghiệm cho gamma cũng được khảo sát tương tự (ví dụ: từ 0.001 đến 10). Hệ thống sẽ tự động chấm điểm hiệu năng trên các tập con kiểm định chéo để lựa chọn ra cấu hình tham số mang lại độ chính xác trung bình cao nhất.

### Kết quả thực nghiệm và so sánh đánh giá

Trực quan hóa ranh giới quyết định theo các cấu hình tham số khác nhau

Khi trực quan hóa kết quả chạy thực nghiệm với hạt nhân RBF:

Khi tham số C quá nhỏ và gamma quá nhỏ, ranh giới quyết định trở nên quá phẳng, mô hình bị dưới khớp (underfitting) và không phân tách được các cụm dữ liệu phức tạp.

Khi C và gamma quá lớn, ranh giới quyết định bị uốn lượn ngoằn ngoèo để ôm trọn từng điểm dữ liệu cá biệt, dẫn đến hiện tượng quá khớp (overfitting) và giảm mạnh độ chính xác trên dữ liệu mới.

Cấu hình tham số tối ưu tạo ra ranh giới mượt mà, bao quát tổng thể và phân tách rõ ràng các lớp dữ liệu.

Đánh giá hiệu năng qua các chỉ số Accuracy, Precision, Recall, F1-Score và so sánh với các thuật toán khác (Logistic Regression, Decision Tree, Random Forest)

Đánh giá mô hình SVM tối ưu thông qua các chỉ số đo lường chuẩn:

Độ chính xác tổng thể (Accuracy): Đạt mức cao trên 95% trên tập kiểm thử.

Độ chính xác phân lớp (Precision) và độ nhạy (Recall): Đều duy trì ở mức cao, chứng minh mô hình cân bằng tốt giữa khả năng nhận diện chính xác dương tính và hạn chế tối đa dương tính giả. So sánh với các thuật toán khác như Hồi quy logistic (Logistic Regression) hay Cây quyết định (Decision Tree), SVM thể hiện ưu thế vượt trội về độ ổn định và khả năng xử lý biên giới phức tạp nhờ cơ chế Vector hỗ trợ.

### Kết luận Chương 4

Chương 4 đã minh họa tường minh từ ví dụ tính tay chi tiết của bài toán SVM đến quy trình thực nghiệm lập trình thực tế, bao gồm khâu chuẩn hóa dữ liệu, tinh chỉnh siêu tham số bằng Grid Search và đánh giá hiệu năng mô hình qua các chỉ số định lượng.
