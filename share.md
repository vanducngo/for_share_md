### **Hướng 1: Explainable AI (XAI) for Diagnosing Customer Segments**

**Tên dự án:** "Diagnosing Customer Dynamics: An Explainable AI Framework for Segment Classification and Transition Analysis"

**Mục tiêu:** Giải thích *tại sao* một khách hàng thuộc về một phân khúc nhất định và *nguyên nhân* gây ra sự chuyển đổi phân khúc.

#### **Pipeline chi tiết:**

**1. Xử lý dữ liệu (Data Processing):**
*   **Input:** Dữ liệu giao dịch thô (CustomerID, InvoiceNo, StockCode, Quantity, Price, InvoiceDate).
*   **Bước 1: Tiền xử lý giao dịch:**
    *   Loại bỏ các giao dịch không hợp lệ (hủy đơn, thiếu CustomerID).
    *   Tạo cột `TotalPrice = Quantity * Price`.
*   **Bước 2: Xây dựng bộ dữ liệu RFM:**
    *   Chọn một ngày chốt sổ (`snapshot_date`).
    *   Tính toán R (Recency), F (Frequency), M (Monetary) cho mỗi khách hàng.
    *   **Kết quả:** Một bảng dữ liệu `customer_rfm` với các cột `[CustomerID, Recency, Frequency, Monetary]`.
*   **Bước 3: Chuẩn bị dữ liệu cho mô hình:**
    *   Xử lý ngoại lệ (IQR method).
    *   Áp dụng Box-Cox transformation để giảm độ xiên (skewness).
    *   Chuẩn hóa dữ liệu bằng `StandardScaler`.
    *   **Kết quả:** Dữ liệu RFM đã được chuẩn hóa, sẵn sàng cho việc phân cụm và phân loại.

**2. Xây dựng Mô hình (Model Building):**
*   **Bước 1: Phân cụm và Gán nhãn (Unsupervised Phase):**
    *   Sử dụng K-Means trên dữ liệu RFM đã chuẩn hóa để phân khách hàng thành K cụm.
    *   Phân tích các centroid của từng cụm để gán nhãn có ý nghĩa kinh doanh (e.g., 'VIP', 'Loyal', 'At-Risk').
    *   Thêm cột `Segment` vào bảng `customer_rfm`.
*   **Bước 2: Xây dựng Mô hình Phân loại (Supervised Phase):**
    *   **Features (X):** `[Recency, Frequency, Monetary]` (đã chuẩn hóa).
    *   **Target (y):** `Segment` (nhãn đã gán).
    *   Chọn một mô hình phân loại mạnh, có thể giải thích được bằng SHAP. **XGBoost** là lựa chọn lý tưởng.
*   **Bước 3: Xây dựng Bộ giải thích (Explainer Building):**
    *   Sử dụng thư viện `shap`.
    *   Khởi tạo một `shap.TreeExplainer` dựa trên mô hình XGBoost đã huấn luyện.

**3. Huấn luyện Mô hình (Model Training):**
*   Chia bộ dữ liệu `customer_rfm` thành tập train (80%) và test (20%).
*   Huấn luyện mô hình XGBoost trên tập train.

**4. Đánh giá Hiệu suất (Performance Evaluation):**
*   **Đánh giá Mô hình Phân loại (Classification Model):**
    *   **Accuracy, Precision, Recall, F1-Score (Macro/Weighted):** Đo lường khả năng của mô hình XGBoost trong việc tái tạo lại các nhãn của K-Means. Mục tiêu là đạt độ chính xác rất cao (>98%) để chứng minh mô hình đã "học" được các ranh giới.
    *   **Confusion Matrix:** Để xem mô hình nhầm lẫn giữa các lớp nào.
*   **Đánh giá Hệ thống Giải thích (Explanation System) - *Đây là phần cốt lõi và mới mẻ***:
    *   **Đánh giá định tính (Qualitative Evaluation):**
        *   **Case Studies:** Chọn các trường hợp khách hàng cụ thể (ví dụ: một khách hàng VIP, một khách hàng At-Risk, một khách hàng vừa chuyển đổi) và trình bày các biểu đồ giải thích (force plot, waterfall plot cho `Delta_SHAP`). Phân tích xem các giải thích này có hợp lý và phù hợp với trực giác kinh doanh không.
    *   **Đánh giá định lượng (Quantitative Evaluation - Nâng cao):**
        *   **Fidelity:** Đo lường mức độ mà lời giải thích của SHAP khớp với dự đoán của mô hình "hộp đen" XGBoost. (Thường thì SHAP có độ trung thực cao với các mô hình cây).
        *   **Stability/Robustness:** Kiểm tra xem một thay đổi nhỏ trong đầu vào có dẫn đến một thay đổi lớn trong lời giải thích không. Lời giải thích tốt nên ổn định.
        *   **(Tùy chọn) User Study:** Thiết kế khảo sát để người dùng (ví dụ: chuyên gia marketing) đánh giá mức độ **dễ hiểu (intelligibility)** và **hữu ích (usefulness)** của các lời giải thích. Đây là thước đo vàng cho một hệ thống XAI.

#### **Đánh giá Độ khả thi (Feasibility Assessment):**

*   **Độ khó kỹ thuật:** **Trung bình.** Các công cụ (XGBoost, SHAP) đã có sẵn và được tài liệu hóa tốt. Thách thức chính nằm ở việc thiết kế phương pháp luận "Differential Explanation" và trình bày kết quả một cách thuyết phục.
*   **Yêu cầu dữ liệu:** **Thấp.** Có thể sử dụng lại bộ dữ liệu từ bài báo trước.
*   **Rủi ro:** **Thấp.** Hướng đi này có rủi ro thất bại thấp vì các công cụ nền tảng đã rất mạnh. Thành công chủ yếu phụ thuộc vào khả năng diễn giải và trình bày kết quả.
*   **Kết luận:** **Rất khả thi.** Đây là một hướng đi an toàn, thời sự và có tiềm năng đóng góp cao.

---

### **Hướng 2: Time Series - Dự báo Giao dịch Tiếp theo**

**Tên dự án:** "When and What Will They Buy Next? A Multi-Task Learning Framework for Predicting Next Purchase Time and Product Category"

**Mục tiêu:** Xây dựng một mô hình duy nhất có thể dự đoán hai việc cùng lúc: (1) **Khi nào** khách hàng sẽ mua hàng lần tiếp theo? (2) **Họ sẽ mua sản phẩm thuộc danh mục nào?**

#### **Pipeline chi tiết:**

**1. Xử lý dữ liệu (Data Processing):**
*   **Input:** Dữ liệu giao dịch thô (CustomerID, InvoiceNo, StockCode, Quantity, Price, InvoiceDate).
*   **Bước 1: Tiền xử lý và Làm giàu dữ liệu:**
    *   Loại bỏ giao dịch không hợp lệ.
    *   Sắp xếp giao dịch của mỗi khách hàng theo `InvoiceDate`.
    *   **Tạo đặc trưng chuỗi thời gian:**
        *   `inter_purchase_time`: Khoảng thời gian (số ngày) giữa hai giao dịch liên tiếp.
        *   `transaction_value`: Tổng giá trị của một giao dịch.
        *   `product_categories`: Danh sách các danh mục sản phẩm trong một giao dịch (cần có một hệ thống phân loại sản phẩm, ví dụ: 'Home Decor', 'Kitchenware').
*   **Bước 2: Xây dựng chuỗi đầu vào (Input Sequences):**
    *   Đối với mỗi khách hàng, tạo ra các chuỗi (sequences) có độ dài cố định `L` (ví dụ: `L=5` giao dịch cuối cùng).
    *   Mỗi phần tử trong chuỗi là một vector đặc trưng của một giao dịch (ví dụ: `[transaction_value, num_items, num_categories, ...]`).
*   **Bước 3: Xây dựng mục tiêu (Target Variables):**
    *   **Target 1 (Regression):** `time_to_next_purchase` (số ngày từ giao dịch cuối cùng trong chuỗi đến giao dịch tiếp theo).
    *   **Target 2 (Classification):** `next_purchase_category` (danh mục sản phẩm chính của giao dịch tiếp theo, dạng one-hot encoded).
    *   **Kết quả:** Một bộ dữ liệu gồm các cặp `(Input_Sequence, [Target1, Target2])`.

**2. Xây dựng Mô hình (Model Building):**
*   **Kiến trúc Multi-Task Learning (MTL):**
    *   **Shared Bottom:** Một hoặc nhiều lớp **LSTM** hoặc **GRU** để học biểu diễn (representation) từ chuỗi giao dịch đầu vào. Lớp này nắm bắt "trạng thái" của khách hàng.
    *   **Task-Specific Heads:**
        *   **Regression Head:** Một vài lớp `Dense` nối tiếp từ Shared Bottom, với lớp cuối cùng có 1 neuron và hàm kích hoạt `linear` (hoặc `relu` để đảm bảo không âm) để dự đoán `time_to_next_purchase`.
        *   **Classification Head:** Một vài lớp `Dense` khác nối tiếp từ Shared Bottom, với lớp cuối cùng có `N` neuron (N là số danh mục sản phẩm) và hàm kích hoạt `softmax` để dự đoán `next_purchase_category`.

**3. Huấn luyện Mô hình (Model Training):**
*   **Loss Function:** Một hàm loss kết hợp. `Total_Loss = w1 * Loss_Regression + w2 * Loss_Classification`.
    *   `Loss_Regression`: Mean Squared Error (MSE) hoặc Mean Absolute Error (MAE).
    *   `Loss_Classification`: Categorical Cross-Entropy.
    *   `w1`, `w2` là các trọng số để cân bằng giữa hai nhiệm vụ.
*   Chia dữ liệu theo thời gian (time-based split) để tránh data leakage. Dùng các giao dịch cũ để huấn luyện và các giao dịch mới hơn để kiểm thử.
*   Huấn luyện mô hình MTL bằng cách tối ưu hóa `Total_Loss`.

**4. Đánh giá Hiệu suất (Performance Evaluation):**
*   **Đánh giá Nhiệm vụ Dự báo Thời gian (Regression Task):**
    *   **Mean Absolute Error (MAE):** "Trung bình, mô hình dự đoán sai lệch bao nhiêu ngày?" (Dễ diễn giải).
    *   **Root Mean Squared Error (RMSE):** Tương tự MAE nhưng trừng phạt các lỗi lớn nặng hơn.
    *   **R-squared (R²):** Đo lường mức độ mà mô hình giải thích được sự biến thiên của dữ liệu.
*   **Đánh giá Nhiệm vụ Dự báo Sản phẩm (Classification Task):**
    *   **Accuracy:** Tỷ lệ dự đoán đúng danh mục.
    *   **Top-k Accuracy:** Tỷ lệ mà danh mục đúng nằm trong top k dự đoán hàng đầu của mô hình (ví dụ: k=3). Điều này hữu ích vì gợi ý một vài sản phẩm cũng đã có giá trị.
    *   **Precision, Recall, F1-Score (per-category và macro/weighted):** Để đánh giá hiệu suất trên từng danh mục.

#### **Đánh giá Độ khả thi (Feasibility Assessment):**

*   **Độ khó kỹ thuật:** **Cao.** Đòi hỏi kiến thức vững về deep learning (RNNs, LSTMs), multi-task learning, và xử lý dữ liệu chuỗi thời gian. Việc chuẩn bị dữ liệu và huấn luyện mô hình phức tạp hơn đáng kể.
*   **Yêu cầu dữ liệu:** **Cao.** Cần dữ liệu giao dịch dày đặc và kéo dài theo thời gian cho mỗi khách hàng để tạo ra các chuỗi có ý nghĩa. Ngoài ra, cần có một hệ thống phân loại sản phẩm (product categorization), thứ không có sẵn trong bộ dữ liệu Online Retail.
*   **Rủi ro:** **Cao.** Mô hình có thể không hội tụ, hoặc một nhiệm vụ có thể lấn át nhiệm vụ kia. Việc tinh chỉnh các trọng số loss và kiến trúc mô hình đòi hỏi nhiều thử nghiệm. Dữ liệu không đủ tốt có thể dẫn đến kết quả kém.
*   **Kết luận:** **Khả thi nhưng đầy thách thức.** Hướng đi này rất tham vọng và có tiềm năng đột phá cao. Tuy nhiên, nó đòi hỏi nền tảng kỹ thuật vững chắc và có thể gặp trở ngại lớn ở khâu chuẩn bị dữ liệu.

---

### **So sánh và Khuyến nghị**

| Tiêu chí | Hướng 1: XAI | Hướng 2: Time Series Prediction |
| :--- | :--- | :--- |
| **Mục tiêu** | Giải thích (WHY) | Dự báo (WHAT & WHEN) |
| **Độ mới** | Cao | Rất cao |
| **Độ khó** | **Trung bình** | **Cao** |
| **Rủi ro** | **Thấp** | **Cao** |
| **Yêu cầu dữ liệu** | **Thấp** | **Cao** |
| **Pipeline** | Tương đối thẳng | Phức tạp, nhiều bước |
| **Tiềm năng đóng góp** | Thực tiễn, dễ áp dụng | Đột phá, giá trị kinh doanh cao |

**Khuyến nghị:**

*   Nếu bạn muốn một dự án có **tỷ lệ thành công cao, rủi ro thấp, và có thể hoàn thành trong một khung thời gian hợp lý**, hãy chọn **Hướng 1: XAI**. Nó xây dựng trực tiếp trên nền tảng bạn đã có và giải quyết một vấn đề rất thời sự.
*   Nếu bạn và nhóm của bạn có **nền tảng deep learning vững chắc, sẵn sàng đối mặt với thách thức kỹ thuật và có thể tìm được nguồn dữ liệu phù hợp** (hoặc tạo ra hệ thống phân loại sản phẩm), hãy chọn **Hướng 2: Time Series Prediction**. Nếu thành công, đây sẽ là một công trình nghiên cứu rất ấn tượng và có sức ảnh hưởng lớn.
