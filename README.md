# 利用爬取化妝品社群評論及標籤 預測產品是否適合消費者 
App 影片網址 https://www.youtube.com/watch?v=gkOtIe6VgcQ
# APP簡介 
使用者輸入膚質資訊及年齡等資訊，預測模型將透過使用者個人資訊預測使用者是否適合此化妝品。

# 負責項目 
透過爬蟲爬取使用者相關資訊，進行資料探勘與預處理以及建立模型。並將預測模型串接到APP上。

# 技術方面
使用Heroku 雲端平台搭配github部署Python的FlaskAPI、預測模型使用python的套件scikit-learn對資料進行訓練並用smote處理資料不平衡的問題、爬蟲使用bs4、資料庫使用mysql。前端使用React-native作呈現。
