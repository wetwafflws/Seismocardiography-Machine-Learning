import sys

filename = "machinelearning_testing.py"
with open(filename, 'r') as f:
    content = f.read()

# Replace checkboxes in __init__
old_checkboxes = """        self.balance_classes_checkbox.setChecked(True)
        self.patient_split_checkbox = QCheckBox("Split by patient (Task I / Task I (MS+AR) / Task I (AS+MR))")
        self.patient_split_checkbox.setToolTip("Keep all segments from a patient in the same split for Task I, Task I (MS+AR), and Task I (AS+MR).")
        self.pause_resume_btn = QPushButton("Pause Training")"""
new_checkboxes = """        self.balance_classes_checkbox.setChecked(True)
        
        self.cv_strategy_dropdown = QComboBox()
        self.cv_strategy_dropdown.addItems(["Standard K-Fold", "Patient-level K-Fold", "Leave-One-Subject-Out (LOSO)"])
        self.cv_strategy_dropdown.setCurrentText("Patient-level K-Fold")
        self.cv_strategy_dropdown.setToolTip("Select the cross-validation strategy.")

        self.augmentation_checkbox = QCheckBox("Enable Time-Series Augmentation")
        self.augmentation_checkbox.setToolTip("Adds random noise and scaling to SCG signals during training to prevent overfitting.")
        self.augmentation_checkbox.setChecked(True)
        
        self.pause_resume_btn = QPushButton("Pause Training")"""
content = content.replace(old_checkboxes, new_checkboxes)

old_layout = """        input_layout.addWidget(self.balance_classes_checkbox)
        input_layout.addWidget(self.patient_split_checkbox)
        input_layout.addWidget(self.step9_btn)"""
new_layout = """        input_layout.addWidget(self.balance_classes_checkbox)
        input_layout.addWidget(self.cv_strategy_dropdown)
        input_layout.addWidget(self.augmentation_checkbox)
        input_layout.addWidget(self.step9_btn)"""
content = content.replace(old_layout, new_layout)

old_enable = """        self.patient_split_checkbox.setEnabled(self.task_selector.currentText() in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)"))"""
new_enable = """        self.cv_strategy_dropdown.setEnabled(self.task_selector.currentText() in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)"))"""
content = content.replace(old_enable, new_enable)

old_task_changed = """        if hasattr(self, 'patient_split_checkbox'):
            is_task1 = task_name in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)")
            self.patient_split_checkbox.setEnabled(is_task1)
            if not is_task1:
                self.patient_split_checkbox.setChecked(False)"""
new_task_changed = """        if hasattr(self, 'cv_strategy_dropdown'):
            is_task1 = task_name in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)")
            self.cv_strategy_dropdown.setEnabled(is_task1)
            if not is_task1:
                self.cv_strategy_dropdown.setCurrentText("Standard K-Fold")"""
content = content.replace(old_task_changed, new_task_changed)

# Update step_train_model
old_split_by_patient = """            split_by_patient = (
                task_name in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)")
                and hasattr(self, 'patient_split_checkbox')
                and self.patient_split_checkbox.isChecked()
            )
            if task_name not in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)") and hasattr(self, 'patient_split_checkbox') and self.patient_split_checkbox.isChecked():
                self.log("[INFO] Patient-level split is only supported for Task I, Task I (MS+AR), and Task I (AS+MR). Using segment-level split.")
                split_by_patient = False

            is_small_training = self.small_training_mode.currentIndex() == 1
            if task_name in ("Task I (MS+AR)", "Task I (AS+MR)"):
                num_epochs = 50
                n_splits = 2
            elif split_by_patient:
                num_epochs = 100
                n_splits = 1
            else:
                num_epochs = 50 if is_small_training else 100
                n_splits = 3 if is_small_training else 5"""
new_split_by_patient = """            cv_mode = "Standard K-Fold"
            if hasattr(self, 'cv_strategy_dropdown'):
                cv_mode = self.cv_strategy_dropdown.currentText()
                if task_name not in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)"):
                    self.log("[INFO] Specific CV strategy is only supported for Task I, Task I (MS+AR), and Task I (AS+MR). Using Standard K-Fold.")
                    cv_mode = "Standard K-Fold"

            is_small_training = self.small_training_mode.currentIndex() == 1
            if task_name in ("Task I (MS+AR)", "Task I (AS+MR)"):
                num_epochs = 50
                n_splits = 2
            elif cv_mode == "Patient-level K-Fold" and task_name in ("Task I", "Task I (MS+AR)", "Task I (AS+MR)"):
                num_epochs = 100
                n_splits = 1
            elif cv_mode == "Leave-One-Subject-Out (LOSO)":
                num_epochs = 50 if is_small_training else 100
                n_splits = -1 # Handled in TrainingWorker
            else:
                num_epochs = 50 if is_small_training else 100
                n_splits = 3 if is_small_training else 5"""
content = content.replace(old_split_by_patient, new_split_by_patient)

old_training_worker_init = """            self.training_worker = TrainingWorker(
                x_tensor=dataset_info['x_tensor'],
                y_tensor=dataset_info['y_tensor'],
                z_tensor=dataset_info['z_tensor'],
                label_tensor=dataset_info['label_tensor'],
                num_classes=5,
                d=64,
                num_epochs=num_epochs,
                batch_size=64,
                learning_rate=0.001,
                weight_decay=0.004,
                test_size=0.2,
                n_splits=n_splits,
                random_state=42,
                multi_label=(task_name == "Task III"),
                patient_ids=dataset_info.get('patient_ids'),
                split_by_patient=split_by_patient,
                class_names=dataset_info.get('class_names'),
            )"""
new_training_worker_init = """            self.training_worker = TrainingWorker(
                x_tensor=dataset_info['x_tensor'],
                y_tensor=dataset_info['y_tensor'],
                z_tensor=dataset_info['z_tensor'],
                label_tensor=dataset_info['label_tensor'],
                num_classes=5,
                d=64,
                num_epochs=num_epochs,
                batch_size=64,
                learning_rate=0.001,
                weight_decay=0.05,
                test_size=0.2,
                n_splits=n_splits,
                random_state=42,
                multi_label=(task_name == "Task III"),
                patient_ids=dataset_info.get('patient_ids'),
                cv_mode=cv_mode,
                use_augmentation=self.augmentation_checkbox.isChecked() if hasattr(self, 'augmentation_checkbox') else False,
                class_names=dataset_info.get('class_names'),
            )"""
content = content.replace(old_training_worker_init, new_training_worker_init)

old_split_log = """            elif split_by_patient:
                self.log(f"[INFO] Split strategy: 80% patient-level train pool / 20% held-out test, then 1-fold (80/20 train/val split) on train pool.")"""
new_split_log = """            elif cv_mode == "Patient-level K-Fold":
                self.log(f"[INFO] Split strategy: 80% patient-level train pool / 20% held-out test, then 1-fold (80/20 train/val split) on train pool.")
            elif cv_mode == "Leave-One-Subject-Out (LOSO)":
                self.log(f"[INFO] Split strategy: Leave-One-Subject-Out (LOSO) on all available patients.")"""
content = content.replace(old_split_log, new_split_log)

with open(filename, 'w') as f:
    f.write(content)

print("UI Patched")
