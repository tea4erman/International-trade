# Загружаем библиотеки

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Выводим на экран информацию о приложении
st.write("В этом приложении вы можете оценить средний срок доставки заказа по заданным характеристикам")

# Шаг 1: Просим пользователя указать значения переменных
avg_customs_days = st.number_input(
    "Введите среднее количество дней на таможне в стране:",
    value = 2,
    placeholder="Введите число...",
    min_value = 1,
    max_value = 7,
    step = 1
)

transport_mode_id = st.selectbox(
    "Выберите тип транспортировки груза",
    ('1', '2', '3', '4', '5', '6'))

# Шаг 2: Предобработка данных для дальнейшей загрузки в модель (если требуется)
features = pd.DataFrame(
    {
        "avg_customs_days": [int(avg_customs_days)], 
        "transport_mode_id": [int(transport_mode_id)]
    }
  )

# Шаг 3: Загружаем модель из сохраненного файла
@st.cache_resource
def load_model(model_path):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), [avg_customs_days]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [transport_mode_id])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    model = joblib.load(model_path)
    return model

model = load_model(r"C:\Users\D\Python scripts\Innopolis 2026\International trade\streamlit_models\delivery_days_model.joblib")

# Шаг 4: Подаем значения переменных в модель и делаем предсказание вероятности y = 1 (инференс)
if st.button("Оценить cрок доставки"):
    y_pred = model.predict(features)

# Шаг 5: Вывод результата на экран
    st.write("Ожидаемый срок доставки составляет %g дней" %np.round(y_pred, 0))