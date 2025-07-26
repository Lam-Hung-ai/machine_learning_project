# Dự đoán rủi do tín dụng tại Đức

### **1. Mục tiêu dự đoán rủi ro tín dụng bằng các mô hình học máy**
**Chủ đề:** Phân loại rủi ro tín dụng (Credit classification)

Trong bài toán này, mục tiêu là xây dựng các mô hình học máy để **dự đoán rủi ro tín dụng** của một người vay ngân hàng, dựa trên các đặc điểm cá nhân và tài chính của họ. Biến mục tiêu là cột `Risk`, với hai giá trị:

* `good`: người có khả năng hoàn trả khoản vay tốt,
* `bad`: người có rủi ro không trả được nợ.

### **2. Giời thiệu bộ dữ liệu**

Nguồn: [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) trên kaggle  
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

### **3. Các mô hình được sử dụng** 🤖

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
### **4. Quy trình thực hiện** 📌

* Tiền xử lý dữ liệu: xử lý giá trị thiếu, mã hóa nhãn (label encoding / one-hot encoding), chuẩn hóa đặc trưng nếu cần thiết.
* Huấn luyện và đánh giá các mô hình: sử dụng các chỉ số như accuracy, precision, recall và F1-score.
* So sánh hiệu suất từng mô hình và chọn ra mô hình tốt nhất để áp dụng cho hệ thống dự đoán rủi ro tín dụng.

### 5. Tiền xử lý dữ liệu

#### 5.1. Khám phá và Dọn dẹp ban đầu

Đây là bước "làm quen" với dữ liệu.
* **Loại bỏ cột không cần thiết**: Code xóa cột `'Unnamed: 0'`. Cột này thường là chỉ số (index) được tạo ra khi lưu file CSV và không mang giá trị dự đoán, vì vậy nó được xem là "nhiễu" và cần được loại bỏ.
* **Xem thống kê cơ bản**: Sử dụng `.info()`, `.describe()`, `.value_counts()` để hiểu về kích thước bộ dữ liệu, các cột chứa gì, kiểu dữ liệu của chúng, và phân phối của các giá trị.

***

#### 5.2. Xử lý Dữ liệu bị thiếu (Missing Data)

Dữ liệu trong thực tế thường không đầy đủ. Code đã xác định được hai cột `Saving accounts` và `Checking account` có giá trị bị thiếu (`NaN`).
* **Phương pháp**: Code sử dụng phương pháp điền vào chỗ trống bằng **mode**.
* **Lý do**: **Mode** là giá trị xuất hiện thường xuyên nhất trong một cột. Đối với các cột dữ liệu định tính (categorical) như "loại tài khoản tiết kiệm", việc điền bằng giá trị phổ biến nhất là một lựa chọn hợp lý và an toàn hơn so với việc đoán mò hoặc xóa dòng dữ liệu (gây mất thông tin).

***

#### 5.3. Xử lý Giá trị Ngoại lai (Outliers)

Giá trị ngoại lai là những điểm dữ liệu khác biệt rất nhiều so với phần còn lại (ví dụ: một người vay số tiền cực lớn so với mặt bằng chung). Những giá trị này có thể làm mô hình bị "lệch" và dự đoán sai.
* **Phương pháp**: Code sử dụng phương pháp **IQR (Interquartile Range)**.
* **Cách hoạt động**: Phương pháp này tính toán một "khoảng giá trị hợp lý" cho mỗi cột số (`Age`, `Credit amount`, `Duration`). Bất kỳ giá trị nào nằm ngoài khoảng này sẽ bị coi là ngoại lai và bị **xóa khỏi bộ dữ liệu**.

***

#### 5.4. Mã hóa và Chuẩn hóa Dữ liệu (Encoding & Standardization) ⚙️

Đây là bước quan trọng nhất, biến đổi dữ liệu thành dạng số mà mô hình có thể tính toán được. Code sử dụng `ColumnTransformer` rất chuyên nghiệp để áp dụng các quy tắc khác nhau cho từng loại cột:

* **Đối với các cột Định tính (Categorical)** như `Sex`, `Job`, `Housing`, `Purpose`:
    * **Hành động**: Áp dụng **One-Hot Encoding**.
    * **Giải thích**: Mô hình không hiểu được chữ ('male', 'female', 'own', 'rent'...). One-Hot Encoding sẽ chuyển mỗi giá trị trong cột này thành một cột nhị phân (0/1) mới. Ví dụ, cột `Housing` với 3 giá trị ('own', 'rent', 'free') sẽ được chuyển thành 3 cột mới: `Housing_own`, `Housing_rent`, `Housing_free`. Nếu một người sở hữu nhà, giá trị ở cột `Housing_own` sẽ là `1` và 2 cột còn lại là `0`.

* **Đối với các cột Số (Numerical)** như `Age`, `Credit amount`, `Duration`:
    * **Hành động**: Áp dụng **StandardScaler**.
    * **Giải thích**: Các cột này có thang đo rất khác nhau (tuổi chỉ vài chục, trong khi số tiền vay có thể lên tới hàng nghìn). Điều này có thể khiến mô hình lầm tưởng rằng `Credit amount` quan trọng hơn `Age`. `StandardScaler` sẽ **chuẩn hóa** tất cả các cột số này về cùng một thang đo (trung bình là 0 và độ lệch chuẩn là 1), giúp mô hình đối xử "công bằng" với tất cả các thuộc tính.

***

### 5. Chia Dữ liệu (Train-Test Split)

Sau khi dữ liệu đã sạch và được chuẩn hóa, bước cuối cùng là chia nó ra.
* **Hành động**: Sử dụng `train_test_split` để chia dữ liệu thành 2 phần:
    * **Tập huấn luyện (Train set)**: Chiếm 80%, dùng để "dạy" cho mô hình học các mẫu (patterns).
    * **Tập kiểm tra (Test set)**: Chiếm 20%, là dữ liệu mà mô hình chưa từng thấy. Nó được dùng để đánh giá xem mô hình hoạt động tốt đến đâu trên dữ liệu thực tế.
* **Tham số quan trọng `stratify=y_encoded`**: Tham số này đảm bảo rằng tỷ lệ hồ sơ "tốt" và "xấu" trong cả tập train và tập test đều **giống hệt nhau** như trong bộ dữ liệu gốc. Điều này cực kỳ quan trọng để việc đánh giá mô hình được khách quan và chính xác.


### **6. Hướng dẫn chạy dự án**
- Tải repo về máy tính. Mở terminal tại folder làm việc gõ lệnh

```cmd
git clone https://github.com/Lam-Hung-ai/machine_learning_project.git
```
- Cài các thư viện cần thiết. Mở terminal tại folder làm việc gõ lệnh
```cmd
pip install -r requirements.txt
```

- chạy toàn bộ file main.ipynb