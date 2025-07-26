# Dự đoán rủi do tín dụng tại Đức

### **1. Mục tiêu dự đoán rủi ro tín dụng bằng các mô hình học máy**

Trong bài toán này, mục tiêu là xây dựng các mô hình học máy để **dự đoán rủi ro tín dụng** của một người vay ngân hàng, dựa trên các đặc điểm cá nhân và tài chính của họ. Biến mục tiêu là cột `Risk`, với hai giá trị:

* `good`: người có khả năng hoàn trả khoản vay tốt,
* `bad`: người có rủi ro không trả được nợ.

### **2. Các mô hình được sử dụng** 🤖

Để giải quyết bài toán phân loại này, tôi áp dụng một số thuật toán học máy phổ biến:

1. **Logistic Regression**

   * Mô hình tuyến tính đơn giản, hiệu quả với các bài toán phân loại nhị phân.
   * Dễ giải thích, thích hợp khi mối quan hệ giữa các biến độc lập và biến mục tiêu là tuyến tính.

2. **K-Nearest Neighbors (KNN)**

   * Thuật toán dựa trên khoảng cách giữa các điểm dữ liệu.
   * Không giả định phân phối dữ liệu, phù hợp với dữ liệu phi tuyến tính.

3. **Decision Tree Classifier**

   * Mô hình cây quyết định đơn giản, dễ hiểu, có khả năng xử lý dữ liệu dạng phân loại và liên tục.
   * Khả năng biểu diễn mối quan hệ phức tạp giữa các đặc trưng.

4. **Stacking Classifier**

   * Mô hình ensemble kết hợp nhiều mô hình con (ở đây có thể gồm Logistic Regression, KNN, và Decision Tree).
   * Mục tiêu là tận dụng ưu điểm của từng mô hình đơn lẻ để cải thiện độ chính xác dự đoán thông qua một mô hình tổng hợp (meta-model).

## **3. Quy trình thực hiện** 📌

* Tiền xử lý dữ liệu: xử lý giá trị thiếu, mã hóa nhãn (label encoding / one-hot encoding), chuẩn hóa đặc trưng nếu cần thiết.
* Huấn luyện và đánh giá các mô hình: sử dụng các chỉ số như accuracy, precision, recall và F1-score.
* So sánh hiệu suất từng mô hình và chọn ra mô hình tốt nhất để áp dụng cho hệ thống dự đoán rủi ro tín dụng.

"German Credit Risk - With Target" trên kaggle [tại đây](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)

**Chủ đề:** Phân loại rủi ro tín dụng (Credit classification)

### **4. Bộ dữ liệu**

Bộ dữ liệu gốc gồm **1000 bản ghi**, được biên soạn bởi **Giáo sư Hofmann**, trong đó **mỗi bản ghi đại diện cho một người vay vốn từ ngân hàng**. Mỗi người được **phân loại là có rủi ro tín dụng "tốt" hoặc "xấu"**, dựa trên tập hợp các thuộc tính có liên quan.

#### 📋 **Nội dung**

* Bộ dữ liệu ban đầu có 20 thuộc tính dạng **categorical/symbolic**.
* Do bộ dữ liệu gốc khá khó hiểu (vì nhiều ký hiệu và phân loại phức tạp), tác giả đã viết một script Python để **chuyển đổi dữ liệu thành định dạng CSV dễ đọc hơn**.
* Một số cột đã bị loại bỏ vì được cho là không cần thiết hoặc khó hiểu.

Dưới đây là phần bảng thuộc tính đã được cập nhật: giữ nguyên tiếng Anh ở phần mô tả kỹ thuật, nhưng cột **"Ý nghĩa"** được ghi bằng **tiếng Việt** để dễ hiểu hơn.

---
#### 📑 **Các thuộc tính trong bộ dữ liệu**

| **Thuộc tính**     | **Kiểu dữ liệu**                                                                                                              | **Ý nghĩa (tiếng Việt)**                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `Age`              | numeric                                                                                                                       | Tuổi của người vay                                                   |
| `Sex`              | text (`male`, `female`)                                                                                                       | Giới tính                                                            |
| `Job`              | numeric:<br>0 - unskilled and non-resident<br>1 - unskilled and resident<br>2 - skilled<br>3 - highly skilled                 | Mức độ công việc và tình trạng cư trú                                |
| `Housing`          | text (`own`, `rent`, `free`)                                                                                                  | Loại hình chỗ ở (sở hữu, thuê, hoặc ở miễn phí)                      |
| `Saving accounts`  | text (`little`, `moderate`, `quite rich`, `rich`)                                                                             | Mức độ tiền tiết kiệm                                                |
| `Checking account` | numeric (Đơn vị: Deutsche Mark - DM)                                                                                          | Số dư tài khoản thanh toán                                           |
| `Credit amount`    | numeric (DM)                                                                                                                  | Số tiền vay tín dụng                                                 |
| `Duration`         | numeric (tháng)                                                                                                               | Thời hạn vay (tính theo tháng)                                       |
| `Purpose`          | text (`car`, `furniture/equipment`, `radio/TV`, `domestic appliances`, `repairs`, `education`, `business`, `vacation/others`) | Mục đích vay tiền                                                    |
| `Risk`             | text (`good`, `bad`)                                                                                                          | **Biến mục tiêu**: đánh giá người vay có rủi ro tín dụng tốt hay xấu |

### **5. Hướng dẫn chạy dự án**

