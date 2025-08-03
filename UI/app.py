import sys
import joblib
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QGroupBox, QScrollArea, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class MedicalForm(QWidget):
    def __init__(self):
        super().__init__()
        self.load_statistics()
        self.initUI()

        self.models = {
            'EQ5D Round 2': {'model': joblib.load('/rds/general/user/hsl121/home/hda_project/hrqol_cv/results/eq5d_round2/models/eq5d_round2_XGB.pkl'), 'type': 'regression'},
            'GAD7 Round 2': {'model': joblib.load('/rds/general/user/hsl121/home/hda_project/hrqol_cv/results/gad7_round2/models/gad7_round2_XGB.pkl'), 'type': 'regression'},
            # 'SQS Round 2': {'model': joblib.load('/rds/general/user/hsl121/home/hda_project/hrqol_cv/results/sqs_round2/models/sqs_round2_Ridge.pkl'), 'type': 'regression'},
            # 'AE Binary': {'model': joblib.load('/rds/general/user/hsl121/home/hda_project/hrqol_cv/results/ae_binary/models/ae_binary_RF.pkl'), 'type': 'classification'},
            # 'AE Severe': {'model': joblib.load('/rds/general/user/hsl121/home/hda_project/hrqol_cv/results/ae_severe/models/ae_severe_RF.pkl'), 'type': 'classification'}
        }

    def load_statistics(self):
        self.numeric_medians = {
            'Age': 45, 'weight': 70, 'height': 165,
            'GAD7_Round1_x': 6, 'EQ5D_Round1': 0.6,
            'insomniaEfficacyMeasure_Round1_x': 5,
            'alcohol_units': 8, 'Smoking_pack_years': 5,
            'Total_THC (mg/g)': 120, 'Total terpene (%w/w)': 0.7
        }

    def initUI(self):
        self.setWindowTitle('PROMs Score Prediction')
        self.resize(1200, 900)
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)
        scrollLayout = QVBoxLayout(scrollContent)

        general_group = QGroupBox("General Info")
        initial_proms_group = QGroupBox("Baseline Scores")
        lifestyle_group = QGroupBox("Lifestyle")
        comorbidity_group = QGroupBox("Comorbidities & Diagnoses")
        prescription_group = QGroupBox("Prescription & Chemistry")

        general_layout = QGridLayout()
        initial_proms_layout = QGridLayout()
        lifestyle_layout = QGridLayout()
        comorbidity_layout = QGridLayout()
        prescription_layout = QGridLayout()

        self.create_input_fields(general_layout, initial_proms_layout, lifestyle_layout, comorbidity_layout, prescription_layout)

        general_group.setLayout(general_layout)
        initial_proms_group.setLayout(initial_proms_layout)
        lifestyle_group.setLayout(lifestyle_layout)
        comorbidity_group.setLayout(comorbidity_layout)
        prescription_group.setLayout(prescription_layout)

        scrollLayout.addWidget(general_group)
        scrollLayout.addWidget(initial_proms_group)
        scrollLayout.addWidget(lifestyle_group)
        scrollLayout.addWidget(comorbidity_group)
        scrollLayout.addWidget(prescription_group)

        scroll.setWidget(scrollContent)
        layout.addWidget(scroll)

        self.submit_button = QPushButton('Predict PROMs Score', self)
        self.submit_button.clicked.connect(self.submit_form)
        layout.addWidget(self.submit_button)

        self.result_label = QLabel('Prediction Results: ', self)
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def create_input_fields(self, gen, base, life, com, presc):
        self.fields = {
            'Age': QLineEdit(self), 'Sex': QComboBox(self), 'occupation': QComboBox(self),
            'weight': QLineEdit(self), 'height': QLineEdit(self),
            'GAD7_Round1_x': QLineEdit(self), 'EQ5D_Round1': QLineEdit(self), 'insomniaEfficacyMeasure_Round1_x': QLineEdit(self),
            'Smoking_status': QComboBox(self), 'Smoking_pack_years': QLineEdit(self),
            'alcohol_units': QLineEdit(self), 'Cannabis_status': QComboBox(self),
            'Myocardial_infarction': QComboBox(self), 'Congestive_heart_failure': QComboBox(self),
            'Peripheral_vascular_disease': QComboBox(self), 'Cerebrovascular_accident_or_transient_ischemic_attack': QComboBox(self),
            'Dementia': QComboBox(self), 'Chronic_obstructive_pulmonary_disease': QComboBox(self),
            'Connective_tissue_disease': QComboBox(self), 'Peptic_Ulcer_Disease': QComboBox(self),
            'Liver_disease': QComboBox(self), 'Diabetes': QComboBox(self), 'Hemiplegia': QComboBox(self),
            'Moderate_to_severe_chronic_kidney_disease': QComboBox(self), 'Solid_tumour': QComboBox(self),
            'Leukemia': QComboBox(self), 'Lymphoma': QComboBox(self), 'AIDS': QComboBox(self),
            'Hypertension': QComboBox(self), 'Depression_or_anxiety': QComboBox(self), 'Arthritis': QComboBox(self),
            'Epilepsy': QComboBox(self), 'VTE': QComboBox(self), 'Endocrine_thyroid_dysfunction': QComboBox(self), 'Allergy': QComboBox(self),
            'form_Capsules': QComboBox(self), 'form_Flos': QComboBox(self), 'form_Oil': QComboBox(self),
            'form_Other': QComboBox(self), 'form_Pastilles': QComboBox(self), 'form_Spray': QComboBox(self),
            'form_Topical': QComboBox(self), 'form_Vape': QComboBox(self),
            'Total_THC (mg/g)': QLineEdit(self), 'Total terpene (%w/w)': QLineEdit(self)
        }

        for field in self.fields:
            if isinstance(self.fields[field], QComboBox):
                self.fields[field].addItems(['0', '1', '2'])

        self.add_fields_to_layout(['Age', 'Sex', 'occupation', 'weight', 'height'], gen)
        self.add_fields_to_layout(['GAD7_Round1_x', 'EQ5D_Round1', 'insomniaEfficacyMeasure_Round1_x'], base)
        self.add_fields_to_layout(['Smoking_status', 'Smoking_pack_years', 'alcohol_units', 'Cannabis_status'], life)
        self.add_fields_to_layout([f for f in self.fields if f.startswith('Myocardial_') or f in ['AIDS', 'Depression_or_anxiety', 'Allergy']], com)
        self.add_fields_to_layout([f for f in self.fields if f.startswith('form_') or 'THC' in f or 'terpene' in f], presc)

    def add_fields_to_layout(self, fields, layout):
        row, col = 0, 0
        for field in fields:
            layout.addWidget(QLabel(field.replace('_', ' ').title()), row, col)
            layout.addWidget(self.fields[field], row, col+1)
            row += 1
            if row % 8 == 0: row, col = 0, col + 2

    def transform_inputs(self, input_data):
        df = pd.DataFrame([input_data])
        df['Sex'] = df['Sex'].replace({'Male': 0, 'Female': 1})
        df['occupation'] = df['occupation'].fillna('Unemployed').map({'Unemployed': 0, 'Retired': 2}).fillna(1)
        binary_map = {'No': 0, 'Yes': 1}
        binary_fields = [k for k in input_data if k in self.fields and isinstance(self.fields[k], QComboBox)]
        for col in binary_fields:
            df[col] = df[col].replace(binary_map).fillna(0)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any() and col in self.numeric_medians:
                df[col].fillna(self.numeric_medians[col], inplace=True)
        return df

    def submit_form(self):
        try:
            input_data = {label: widget.currentText() if isinstance(widget, QComboBox) else widget.text()
                          for label, widget in self.fields.items()}
            input_df = self.transform_inputs(input_data)
            results = {}
            for name, content in self.models.items():
                model, model_type = content['model'], content['type']
                prediction = model.predict(input_df)[0]
                if model_type == 'classification':
                    prob = model.predict_proba(input_df)[0][1]
                    results[name] = f"{'Yes' if prediction == 1 else 'No'} (Prob: {prob:.2%})"
                else:
                    results[name] = round(prediction, 3)
            result_text = '\n'.join([f"{k}: {v}" for k, v in results.items()])
            self.result_label.setText(f"Prediction Results:\n{result_text}")
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to make predictions: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MedicalForm()
    window.show()
    sys.exit(app.exec_())
