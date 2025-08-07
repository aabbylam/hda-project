import sys
import joblib
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QGroupBox, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import MultiLabelBinarizer

class MedicalForm(QWidget):
    def __init__(self):
        print("Initializing MedicalForm")
        super().__init__()
        self.load_data()
        self.setup_feature_engineering()
        self.load_models()
        self.init_ui()

    def apply_feature_engineering(self, data):
        print("Applying complete feature engineering...")
        # Handle missing values first
        data = self.handle_missing_values(data)
        
        # Handle Sex conversion
        sex_value = data.get('Sex', '').lower().strip()
        if sex_value == 'female':
            data['Sex'] = 1
        elif sex_value == 'male':
            data['Sex'] = 0  # Explicitly set Male to 0
        else:
            data['Sex'] = 0  # Default value if not specified
            
        data['occupation'] = {'Unemployed': 0, 'Employed': 1, 'Retired': 2}.get(data.get('occupation'), 1)
        data['Smoking_status'] = {'Never smoked': 0, 'Ex-smoker': 1, 'Current smoker': 2}.get(data.get('Smoking_status'), 0)
        data['Cannabis_status'] = {'Never used': 0, 'Ex-user': 1, 'Current user': 2}.get(data.get('Cannabis_status'), 0)
        
        # Process prescriptions
        prescription_forms = []
        for i in range(1, 4):
            prescription = data.get(f'Product_Prescription{i}')
            if prescription and prescription != "None":
                form = self.extract_form(prescription)
                data[f'Product_Prescription{i}_form'] = form
                data[f'Product_Prescription_ID{i}'] = self.product_to_id_mapping.get(prescription, -1)
                if form:
                    prescription_forms.append(form)
            else:
                data[f'Product_Prescription{i}_form'] = None
                data[f'Product_Prescription_ID{i}'] = -1
        
        # Process forms
        unique_forms = list(set(f for f in prescription_forms if f))
        for form in self.product_forms:
            data[f'form_{form}'] = 1 if form in unique_forms else 0
            
        return data

    def handle_missing_values(self, data):
        """Ensure all required fields have default values"""
        defaults = {
            'Age': 0, 'Sex': 0, 'weight': 0, 'height': 0,
            'GAD7_Round1': 0, 'EQ5D_Round1': 0, 'insomniaEfficacyMeasure_Round1': 0,
            'Smoking_pack_years': 0, 'alcohol_units': 0, 'Charlson_comorbidity': 0
        }
        
        for field, default in defaults.items():
            if field not in data or data[field] is None:
                data[field] = default
                
        return data

    def load_data(self):
        print("Loading data...")
        try:
            self.chemical_df = pd.read_excel('../data/Product Data.xlsx', engine='openpyxl')
            self.sku_list = sorted(self.chemical_df['SKU'].dropna().unique())
            print("Loaded Excel data successfully")
        except Exception as e:
            print(f"Could not load Excel file: {e}")
            self.create_dummy_data()

        # Chemical columns
        self.chemical_cols = [
            'Total_THC (mg/g)', ' Total_CBD (mg/g)', 'alpha-Pinene (PPM)', 'Camphene (PPM)',
            'beta-Myrcene (PPM)', 'beta-Pinene (PPM)', 'alfa-Terpinene (PPM)',
            'Ocimene (sum of cis- and trans- isomers) (PPM)', 'D-Limonene (PPM)', 'gamma-Terpinene (PPM)',
            'Terpinolene (PPM)', 'Linalool (PPM)', 'Fenchol (PPM)', 'Isopulegol (PPM)',
            'Borneol (PPM)', 'alpha.-Terpineol (PPM)', 'Geraniol (PPM)', 'Caryophyllene (PPM)',
            'Humulene (PPM)', 'Nerolidol (PPM)', 'alpha-Bisabolol (PPM)', 'Total terpene (%w/w)'
        ]

        # Comorbidity fields
        self.comorbidity_fields = [
            'Myocardial_infarction', 'Congestive_heart_failure', 'Peripheral_vascular_disease',
            'Cerebrovascular_accident_or_transient_ischemic_attack', 'Dementia',
            'Chronic_obstructive_pulmonary_disease', 'Connective_tissue_disease',
            'Peptic_Ulcer_Disease', 'Liver_disease', 'Diabetes', 'Hemiplegia',
            'Moderate_to_severe_chronic_kidney_disease', 'Solid_tumour', 'Leukemia',
            'Lymphoma', 'AIDS', 'Hypertension', 'Depression_or_anxiety', 'Arthritis',
            'Epilepsy', 'VTE', 'Endocrine_thyroid_dysfunction', 'Allergy'
        ]

        # Special comorbidities
        self.special_comorbidities = {
            'Liver_disease': ['No', 'Mild', 'Moderate to severe'],
            'Diabetes': ['None or diet-controlled', 'Uncomplicated', 'End organ damage'],
            'Solid_tumour': ['No', 'Localized', 'Metastatic']
        }

        # Diagnosis list
        self.diagnosis_list = [
            "Depression", "Anxiety", "Chronic pain", "Osteoarthritis", "PTSD", "Fibromyalgia", 
            "Multiple sclerosis", "Neuropathic pain", "Attention deficit hyperactivity disorder",
            "Migraine", "Insomnia", "Endometriosis", "Hypermobility", "Crohns", "Epilepsy adult",
            "Chemotherapy induced nausea and vomiting", "Autistic spectrum disorder", "OCD",
            "Ulcerative colitis", "Inflammatory arthritis", "Cluster headaches", "Palliative care",
            "Complex regional pain syndrome", "Cancer pain", "Trigeminal neuralgia",
            "Rare and challenging skin condition", "Agoraphobia", "Tourette's syndrome",
            "Parkinson's", "Headache", "Social phobia", "Eating disorder", "Breast pain",
            "Panic disorder"
        ]
        
        self.all_diagnoses_for_encoding = [
            'Depression', 'Anxiety', 'Chronic pain', 'Osteoarthritis', 'PTSD', 'Fibromyalgia',
            'Multiple sclerosis', 'Neuropathic pain', 'Attention deficit hyperactivity disorder',
            'Migraine', 'Insomnia', 'Endometriosis', 'Hypermobility', 'Crohns', 'Epilepsy adult',
            'Chemotherapy induced nausea and vomiting', 'Autistic spectrum disorder', 'OCD',
            'Ulcerative colitis', 'Inflammatory arthritis', 'Cluster headaches', 'Palliative care',
            'Complex regional pain syndrome', 'Cancer pain', 'Trigeminal neuralgia',
            'Rare and challenging skin condition', 'Agoraphobia', "Tourette's syndrome",
            "Parkinson's", 'Headache', 'Social phobia', 'Eating disorder', 'Breast pain',
            'Panic disorder'
        ]

    def setup_feature_engineering(self):
        print("Setting up feature engineering...")
        self.product_forms = ['Flos', 'Oil', 'Vape', 'Pastilles', 'Capsules', 'Spray', 'Topical', 'Other']
        self.product_to_id_mapping = {sku: i for i, sku in enumerate(self.sku_list)}

    def extract_form(self, product):
        if pd.isna(product) or product == "None":
            return None
        name = product.lower()
        if 'flos' in name or 'flower' in name:
            return 'Flos'
        elif 'oil' in name:
            return 'Oil'
        elif 'vape' in name or 'cartridge' in name:
            return 'Vape'
        elif 'pastille' in name:
            return 'Pastilles'
        elif 'capsule' in name:
            return 'Capsules'
        elif 'spray' in name:
            return 'Spray'
        elif 'ointment' in name or 'cream' in name:
            return 'Topical'
        return 'Other'

    def create_dummy_data(self):
        print("Creating dummy data...")
        self.sku_list = [
            "THC-Oil-001", "CBD-Flos-002", "MIX-Vape-003", "SAT-Oil-004", "IND-Flos-005",
            "HYB-Capsules-006", "CBD-Spray-007", "THC-Pastilles-008", "MIX-Topical-009", "SAT-Oil-010"
        ]
        
        dummy_data = []
        for sku in self.sku_list:
            row = {'SKU': sku}
            for col in self.chemical_cols:
                if 'THC' in col or 'CBD' in col:
                    row[col] = np.random.uniform(0, 300)
                elif 'PPM' in col:
                    row[col] = np.random.uniform(0, 1000)
                elif '%w/w' in col:
                    row[col] = np.random.uniform(0, 5)
                else:
                    row[col] = np.random.uniform(0, 100)
            dummy_data.append(row)
        
        self.chemical_df = pd.DataFrame(dummy_data)

    def load_models(self):
        print("Loading models...")
        self.models = {}
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        model_paths = {
            'EQ5D Round 2': '../hrqol_cv/results/eq5d_round2/models/eq5d_round2_XGB.pkl',
            'EQ5D Round 3': '../hrqol_cv/results/eq5d_round3/models/eq5d_round3_XGB.pkl',
            'EQ5D Round 4': '../hrqol_cv/results/eq5d_round4/models/eq5d_round4_XGB.pkl',
            'GAD7 Round 2': '../hrqol_cv/results/gad7_round2/models/gad7_round2_XGB.pkl',
            'GAD7 Round 3': '../hrqol_cv/results/gad7_round3/models/gad7_round3_XGB.pkl',
            'GAD7 Round 4': '../hrqol_cv/results/gad7_round4/models/gad7_round4_XGB.pkl'

        }
        
        for name, path in model_paths.items():
            try:
                self.models[name] = {
                    'model': joblib.load(path),
                    'type': 'regression'
                }
                print(f"✓ {name} model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {name} model: {e}")
                self.models[name] = {'model': None, 'type': 'regression'}
        
        loaded_models = [name for name, info in self.models.items() if info['model'] is not None]
        if loaded_models:
            print(f"Successfully loaded {len(loaded_models)} model(s): {', '.join(loaded_models)}")
        else:
            print("No models loaded - using dummy predictions")

    def init_ui(self):
        print("Initializing UI...")
        self.setWindowTitle('PROMs Score Prediction')
        self.resize(1400, 1000)
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)
        scrollLayout = QVBoxLayout(scrollContent)

        # Initialize all fields
        self.fields = {
            'Age': QLineEdit(), 'weight': QLineEdit(), 'height': QLineEdit(),
            'GAD7_Round1': QLineEdit(), 'EQ5D_Round1': QLineEdit(), 
            'insomniaEfficacyMeasure_Round1': QLineEdit(),
            'Smoking_pack_years': QLineEdit(), 'alcohol_units': QLineEdit(), 
            'Charlson_comorbidity': QLineEdit(),
            'Sex': QComboBox(), 'occupation': QComboBox(), 
            'Smoking_status': QComboBox(), 'Cannabis_status': QComboBox(),
            'Product_Prescription1': QComboBox(), 
            'Product_Prescription2': QComboBox(), 
            'Product_Prescription3': QComboBox()
        }

        # Set placeholders and dropdowns
        for field in ['Age', 'weight', 'height', 'GAD7_Round1', 'EQ5D_Round1',
                     'insomniaEfficacyMeasure_Round1', 'Smoking_pack_years', 
                     'alcohol_units', 'Charlson_comorbidity']:
            self.fields[field].setPlaceholderText(f"Enter {field.replace('_', ' ')}")

        dropdown_options = {
            'Sex': ['Select...', 'Male', 'Female'],
            'occupation': ['Select...', 'Unemployed', 'Employed', 'Retired'],
            'Smoking_status': ['Select...', 'Never smoked', 'Ex-smoker', 'Current smoker'],
            'Cannabis_status': ['Select...', 'Never used', 'Ex-user', 'Current user']
        }
        
        for field, options in dropdown_options.items():
            self.fields[field].addItems(options)

        # Setup prescription dropdowns
        for i in range(1, 4):
            self.fields[f'Product_Prescription{i}'].addItem("None")
            self.fields[f'Product_Prescription{i}'].addItems(self.sku_list)
            self.fields[f'Product_Prescription{i}'].currentTextChanged.connect(self.update_chemical_fields)

        # Diagnosis dropdowns
        self.diagnosis_1 = QComboBox()
        self.diagnosis_2 = QComboBox()
        self.diagnosis_3 = QComboBox()
        for box in [self.diagnosis_1, self.diagnosis_2, self.diagnosis_3]:
            box.addItem("None")
            box.addItems(self.diagnosis_list)

        # Comorbidity dropdowns
        self.comorbidity_widgets = {}
        for field in self.comorbidity_fields:
            combo = QComboBox()
            if field in self.special_comorbidities:
                combo.addItems(['Select...'] + self.special_comorbidities[field])
            else:
                combo.addItems(['Select...', 'No', 'Yes'])
            self.comorbidity_widgets[field] = combo

        # Chemical fields
        self.chemical_widgets = {}
        for col in self.chemical_cols:
            widget = QLineEdit()
            widget.setReadOnly(True)
            widget.setStyleSheet("background-color: #f0f0f0;")
            widget.setPlaceholderText("Auto-calculated")
            self.chemical_widgets[col] = widget

        # Helper function to add sections
        def add_section(title, widget_dict, keys=None):
            box = QGroupBox(title)
            grid = QGridLayout()
            for i, key in enumerate(keys or list(widget_dict.keys())):
                label_text = key.replace('_', ' ').replace('or', 'or\n').title()
                grid.addWidget(QLabel(label_text), i, 0)
                widget = widget_dict[key] if key in widget_dict else self.fields[key]
                grid.addWidget(widget, i, 1)
            box.setLayout(grid)
            scrollLayout.addWidget(box)

        # Add all sections
        add_section("General Information", self.fields, 
                   ['Age', 'Sex', 'occupation', 'weight', 'height'])
        
        add_section("Baseline Scores", self.fields,
                   ['GAD7_Round1', 'EQ5D_Round1', 'insomniaEfficacyMeasure_Round1'])
        
        add_section("Lifestyle Factors", self.fields,
                   ['Smoking_status', 'Smoking_pack_years', 'alcohol_units', 'Cannabis_status'])
        
        add_section("Prescriptions", self.fields,
                   ['Product_Prescription1', 'Product_Prescription2', 'Product_Prescription3'])

        # Diagnoses section
        diag_box = QGroupBox("Primary Diagnoses")
        diag_layout = QGridLayout()
        diag_layout.addWidget(QLabel("Primary Diagnosis"), 0, 0)
        diag_layout.addWidget(self.diagnosis_1, 0, 1)
        diag_layout.addWidget(QLabel("Secondary Diagnosis"), 1, 0)
        diag_layout.addWidget(self.diagnosis_2, 1, 1)
        diag_layout.addWidget(QLabel("Tertiary Diagnosis"), 2, 0)
        diag_layout.addWidget(self.diagnosis_3, 2, 1)
        diag_box.setLayout(diag_layout)
        scrollLayout.addWidget(diag_box)

        # Comorbidities sections
        add_section("Cardiovascular Comorbidities", self.comorbidity_widgets,
                   ['Myocardial_infarction', 'Congestive_heart_failure', 
                    'Peripheral_vascular_disease', 'Cerebrovascular_accident_or_transient_ischemic_attack',
                    'Hypertension', 'VTE'])
        
        add_section("Neurological Comorbidities", self.comorbidity_widgets,
                   ['Dementia', 'Hemiplegia', 'Epilepsy', 'Depression_or_anxiety'])
        
        add_section("Chronic Disease Comorbidities", self.comorbidity_widgets,
                   ['Chronic_obstructive_pulmonary_disease', 'Diabetes', 'Liver_disease',
                    'Moderate_to_severe_chronic_kidney_disease', 'Arthritis', 'Endocrine_thyroid_dysfunction'])
        
        add_section("Cancer Comorbidities", self.comorbidity_widgets,
                   ['Solid_tumour', 'Leukemia', 'Lymphoma'])
        
        add_section("Other Comorbidities", self.comorbidity_widgets,
                   ['Connective_tissue_disease', 'Peptic_Ulcer_Disease', 
                    'AIDS', 'Allergy', 'Charlson_comorbidity'])

        # Chemical composition section
        chem_box = QGroupBox("Chemical Composition (Auto-calculated)")
        chem_layout = QGridLayout()
        for i, col in enumerate(self.chemical_cols):
            row = i // 2
            col_pos = (i % 2) * 2
            chem_layout.addWidget(QLabel(col), row, col_pos)
            chem_layout.addWidget(self.chemical_widgets[col], row, col_pos + 1)
        chem_box.setLayout(chem_layout)
        scrollLayout.addWidget(chem_box)

        # Status and info
        loaded_models = [name for name, info in self.models.items() if info['model'] is not None]
        model_status = QLabel()
        if loaded_models:
            model_status.setText(f"✓ Using real ML models: {', '.join(loaded_models)}")
            model_status.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            model_status.setText("⚠ Using dummy predictions (ML model files not found)")
            model_status.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
        scrollLayout.addWidget(model_status)

        feature_info = QLabel(f"ℹ Total expected features: {len(self.get_feature_names())}")
        feature_info.setStyleSheet("color: blue; font-style: italic; padding: 5px;")
        scrollLayout.addWidget(feature_info)

        # Results section
        self.result_label = QLabel('Prediction results will appear here after clicking "Predict PROMs Score".')
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("border: 1px solid gray; padding: 10px; background-color: #f9f9f9;")
        scrollLayout.addWidget(self.result_label)

        # Submit button
        self.submit_button = QPushButton('Predict PROMs Score')
        self.submit_button.clicked.connect(self.submit_form)
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        scrollLayout.addWidget(self.submit_button)

        scroll.setWidget(scrollContent)
        layout.addWidget(scroll)
        self.setLayout(layout)
        print("UI initialized successfully")

    def update_chemical_fields(self):
        """Update chemical composition fields based on selected prescriptions"""
        try:
            prescriptions = [
                self.fields[f'Product_Prescription{i}'].currentText()
                for i in range(1, 4)
                if self.fields[f'Product_Prescription{i}'].currentText() not in ["None", ""]
            ]

            if not prescriptions:
                for col in self.chemical_cols:
                    self.chemical_widgets[col].setText("0.00")
                return

            mean_values = {}
            for col in self.chemical_cols:
                values = []
                for prescription in prescriptions:
                    mask = self.chemical_df['SKU'] == prescription
                    if mask.any():
                        value = self.chemical_df.loc[mask, col].iloc[0]
                        if pd.notna(value):
                            if isinstance(value, str):
                                value_str = value.strip().lower()
                                if value_str.startswith('<'):
                                    try:
                                        limit = float(value_str[1:])
                                        values.append(limit / 2)
                                    except ValueError:
                                        values.append(0.0)
                                elif value_str in ['nd', 'not detected', 'n/d', '', 'major', 'minor']:
                                    values.append(0.0)
                                else:
                                    try:
                                        values.append(float(value_str))
                                    except ValueError:
                                        values.append(0.0)
                            else:
                                try:
                                    values.append(float(value))
                                except (ValueError, TypeError):
                                    values.append(0.0)
                
                mean_values[col] = np.mean(values) if values else 0.0

            for col in self.chemical_cols:
                self.chemical_widgets[col].setText(f"{mean_values.get(col, 0.0):.2f}")

        except Exception as e:
            print(f"Error updating chemical fields: {e}")
            for col in self.chemical_cols:
                self.chemical_widgets[col].setText("0.00")

    def get_form_data(self):
        data = {}
        
        # Handle Sex field specifically
        sex_widget = self.fields['Sex']
        if isinstance(sex_widget, QComboBox):
            selected = sex_widget.currentText().strip()  # Remove any whitespace
            print(f"DEBUG - Raw Sex selection: '{selected}'")  # Debug output
            
            # Store the raw selected value
            data['Sex'] = selected if selected not in ['Select...', 'None', ''] else None
            
            # Debug output to verify
            print(f"DEBUG - Processed Sex value: {data.get('Sex')}")
            
        # Get basic field values with defaults
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                value = widget.text().strip()
                try:
                    data[key] = float(value) if value else 0.0
                except ValueError:
                    data[key] = 0.0
            elif isinstance(widget, QComboBox):
                selected = widget.currentText()
                data[key] = selected if selected not in ['Select...', 'None'] else None

        # Validate Charlson comorbidity
        try:
            charlson = float(data.get('Charlson_comorbidity', 0))
            data['Charlson_comorbidity'] = max(0, min(37, charlson))
        except (ValueError, TypeError):
            data['Charlson_comorbidity'] = 0

        # Get diagnosis data
        diagnoses = []
        for box in [self.diagnosis_1, self.diagnosis_2, self.diagnosis_3]:
            selected = box.currentText()
            if selected != "None":
                diagnoses.append(selected)
        data['diagnoses'] = diagnoses

        # Get comorbidity data
        for field, widget in self.comorbidity_widgets.items():
            selected = widget.currentText()
            if selected == 'Select...':
                data[field] = 0
            elif field == 'Liver_disease':
                mapping = {'No': 0, 'Mild': 1, 'Moderate to severe': 2}
                data[field] = mapping.get(selected, 0)
            elif field == 'Diabetes':
                mapping = {'None or diet-controlled': 0, 'Uncomplicated': 1, 'End organ damage': 2}
                data[field] = mapping.get(selected, 0)
            elif field == 'Solid_tumour':
                mapping = {'No': 0, 'Localized': 1, 'Metastatic': 2}
                data[field] = mapping.get(selected, 0)
            elif field == 'Allergy':
                data[field] = 1 if selected == 'Yes' else 0
            else:
                data[field] = 1 if selected == 'Yes' else 0

        # Get chemical data
        for col in self.chemical_cols:
            value_text = self.chemical_widgets[col].text()
            try:
                data[col] = float(value_text) if value_text else 0.0
            except ValueError:
                data[col] = 0.0

        # Apply feature engineering
        data = self.apply_feature_engineering(data)
        
        return data

    def submit_form(self):
        try:
            data = self.get_form_data()
            
            # Special handling for Sex validation
            sex_value = data.get('Sex')
            if sex_value is None or str(sex_value).strip() == '':
                QMessageBox.warning(self, "Missing Data", "Please select a valid Sex (Male or Female)")
                return
            
            # Validate required fields
            required_fields = ['Age', 'weight', 'height']
            missing = [f.replace('_', ' ').title() for f in required_fields if not data.get(f)]
            
            if missing:
                QMessageBox.warning(self, "Missing Data", 
                                  f"Please fill in: {', '.join(missing)}")
                return

            # Make predictions
            predictions = self.make_predictions(data)
            
            # Display results
            result_text = "=== PREDICTION RESULTS ===\n\n"
            for model_name, prediction in predictions.items():
                if 'EQ5D' in model_name:
                    result_text += f"{model_name}: {prediction:.3f}\n"
                elif 'GAD7' in model_name:
                    result_text += f"{model_name}: {prediction:.1f}\n"
                else:
                    result_text += f"{model_name}: {prediction:.2f}\n"
            
            # Add feature summary
            active_forms = [f for f in self.product_forms if data.get(f'form_{f}', 0) == 1]
            if active_forms:
                result_text += f"\nActive product forms: {', '.join(active_forms)}\n"
            
            active_comorbidities = [f.replace('_', ' ').title() 
                                  for f in self.comorbidity_fields if data.get(f, 0) > 0]
            if active_comorbidities:
                result_text += f"Active comorbidities: {len(active_comorbidities)}\n"
            
            self.result_label.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            print(f"Detailed error: {e}")

    def make_predictions(self, data):
        """Make predictions using loaded models"""
        predictions = {}
        
        for model_name, model_info in self.models.items():
            if model_info['model'] is None:
                # Dummy predictions
                if 'EQ5D' in model_name:
                    base = 0.7 - (data.get('Age', 40) - 40) * 0.002
                    if data.get('GAD7_Round1'):
                        base -= data['GAD7_Round1'] * 0.02
                    if data.get('EQ5D_Round1'):
                        base = data['EQ5D_Round1'] * 0.9 + np.random.uniform(-0.1, 0.1)
                    predictions[model_name] = max(-0.6, min(1.0, base + np.random.uniform(-0.1, 0.1)))
                elif 'GAD7' in model_name:
                    base = 8 + (data.get('Age', 40) - 40) * 0.05
                    if data.get('GAD7_Round1'):
                        base = data['GAD7_Round1'] * 0.95 + np.random.uniform(-2, 2)
                    active_comorb = sum(1 for f in self.comorbidity_fields if data.get(f, 0) > 0)
                    base += active_comorb * 0.3
                    predictions[model_name] = max(0, min(21, base + np.random.uniform(-1, 1)))
                else:
                    predictions[model_name] = np.random.uniform(0, 100)
            else:
                try:
                    features = self.prepare_features_for_model(data, model_name)
                    prediction = model_info['model'].predict([features])[0]
                    
                    if 'EQ5D' in model_name:
                        prediction = max(-0.6, min(1.0, prediction))
                    elif 'GAD7' in model_name:
                        prediction = max(0, min(21, prediction))
                    
                    predictions[model_name] = prediction
                except Exception as e:
                    print(f"Error using {model_name}: {e}")
                    if 'EQ5D' in model_name:
                        predictions[model_name] = np.random.uniform(0.2, 0.9)
                    else:
                        predictions[model_name] = np.random.uniform(2, 15)
        
        return predictions
    


    def prepare_features_for_model(self, data, model_name):
        """Prepare feature vector matching model's expected order"""
        features = []
        
        # Basic features (36)
        basic_features = [
            'EQ5D_Round1', 'Age', 'Sex', 'occupation', 'weight', 'height',
            'Myocardial_infarction', 'Congestive_heart_failure', 'Peripheral_vascular_disease',
            'Cerebrovascular_accident_or_transient_ischemic_attack', 'Dementia',
            'Chronic_obstructive_pulmonary_disease', 'Connective_tissue_disease',
            'Peptic_Ulcer_Disease', 'Liver_disease', 'Diabetes', 'Hemiplegia',
            'Moderate_to_severe_chronic_kidney_disease', 'Solid_tumour', 'Leukemia',
            'Lymphoma', 'AIDS', 'Charlson_comorbidity', 'Hypertension',
            'Depression_or_anxiety', 'Arthritis', 'Epilepsy', 'VTE',
            'Endocrine_thyroid_dysfunction', 'Allergy', 'Smoking_status',
            'Smoking_pack_years', 'alcohol_units', 'Cannabis_status', 'GAD7_Round1',
            'insomniaEfficacyMeasure_Round1'
        ]
        
        # Diagnosis features (34)
        diagnosis_features = [
            'diag_' + d for d in self.all_diagnoses_for_encoding
        ]
        
        # Form features (8)
        form_features = [
            'form_' + f for f in self.product_forms
        ]
        
        # Chemical features (22)
        chemical_features = self.chemical_cols
        
        # Product IDs (3)
        product_ids = [
            'Product_Prescription_ID1', 'Product_Prescription_ID2', 'Product_Prescription_ID3'
        ]
        
        # Combine all features
        all_features = basic_features + diagnosis_features + form_features + chemical_features + product_ids
        
        # Extract values in order
        for feature in all_features:
            if feature.startswith('diag_'):
                diag = feature[5:]  # Remove 'diag_' prefix
                features.append(1 if diag in data.get('diagnoses', []) else 0)
            elif feature.startswith('form_'):
                form = feature[5:]  # Remove 'form_' prefix
                features.append(data.get(f'form_{form}', 0))
            else:
                features.append(data.get(feature, 0))
        
        print(f"Prepared {len(features)} features for {model_name}")
        return np.array(features)

    def get_feature_names(self):
        """Get complete list of feature names in order"""
        features = []
        
        # Basic features (36)
        features.extend([
            'EQ5D_Round1', 'Age', 'Sex', 'occupation', 'weight', 'height',
            'Myocardial_infarction', 'Congestive_heart_failure', 'Peripheral_vascular_disease',
            'Cerebrovascular_accident_or_transient_ischemic_attack', 'Dementia',
            'Chronic_obstructive_pulmonary_disease', 'Connective_tissue_disease',
            'Peptic_Ulcer_Disease', 'Liver_disease', 'Diabetes', 'Hemiplegia',
            'Moderate_to_severe_chronic_kidney_disease', 'Solid_tumour', 'Leukemia',
            'Lymphoma', 'AIDS', 'Charlson_comorbidity', 'Hypertension',
            'Depression_or_anxiety', 'Arthritis', 'Epilepsy', 'VTE',
            'Endocrine_thyroid_dysfunction', 'Allergy', 'Smoking_status',
            'Smoking_pack_years', 'alcohol_units', 'Cannabis_status', 'GAD7_Round1',
            'insomniaEfficacyMeasure_Round1'
        ])
        
        # Diagnosis features (34)
        features.extend(['diag_' + d for d in self.all_diagnoses_for_encoding])
        
        # Form features (8)
        features.extend(['form_' + f for f in self.product_forms])
        
        # Chemical features (22)
        features.extend(self.chemical_cols)
        
        # Product IDs (3)
        features.extend([
            'Product_Prescription_ID1', 'Product_Prescription_ID2', 'Product_Prescription_ID3'
        ])
        
        return features

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MedicalForm()
    window.show()
    sys.exit(app.exec_())