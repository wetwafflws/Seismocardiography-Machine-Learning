import sys

filename = "machinelearning_testing.py"
with open(filename, 'r') as f:
    content = f.read()

# Replace focal loss
content = content.replace("criterion = nn.CrossEntropyLoss(weight=fold_class_weights.to(device))", "criterion = FocalLoss(weight=fold_class_weights.to(device), gamma=2.0)")

# Add augmentation
old_aug_target = """                        if self.multi_label:
                            labels = labels.float()"""
new_aug_target = """                        if self.use_augmentation and model.training:
                            noise_std = 0.05
                            batch_x = batch_x + torch.randn_like(batch_x) * noise_std
                            batch_y = batch_y + torch.randn_like(batch_y) * noise_std
                            batch_z = batch_z + torch.randn_like(batch_z) * noise_std
                            scale = torch.empty(batch_x.size(0), 1, 1).uniform_(0.8, 1.2).to(device)
                            batch_x = batch_x * scale
                            batch_y = batch_y * scale
                            batch_z = batch_z * scale
                            
                        if self.multi_label:
                            labels = labels.float()"""
content = content.replace(old_aug_target, new_aug_target)

# Update splitting logic
old_split_logic_start = "if self.split_by_patient:"
old_split_logic_end = "            else:\n                self.log_update.emit(f\"80/20 split done: train={len(train_idx)} samples, test={len(test_idx)} samples\")"

idx_start = content.find(old_split_logic_start)
idx_end = content.find(old_split_logic_end) + len(old_split_logic_end)

new_split_logic = """is_patient_level = self.cv_mode in ("Patient-level K-Fold", "Leave-One-Subject-Out (LOSO)")
            
            if is_patient_level:
                if self.multi_label:
                    raise RuntimeError("Patient-level splitting is not supported for Task III.")
                if self.patient_ids is None:
                    raise RuntimeError("Patient IDs are required for patient-level splitting.")

                patient_ids_np = np.asarray(self.patient_ids)
                if len(patient_ids_np) != len(labels_np):
                    raise RuntimeError("Patient IDs length does not match label tensor length.")

                patient_to_label = {}
                patient_to_indices = {}
                for idx, patient_id in enumerate(patient_ids_np):
                    if patient_id not in patient_to_indices:
                        patient_to_indices[patient_id] = []
                    patient_to_indices[patient_id].append(idx)
                    if patient_id not in patient_to_label:
                        patient_to_label[patient_id] = int(labels_np[idx])

                patients = np.array(list(patient_to_label.keys()))
                patient_labels = np.array([patient_to_label[pid] for pid in patients])
                class_name_map = self.class_names or [str(i) for i in range(self.num_classes)]
                unique_patients, patient_counts = np.unique(patient_labels, return_counts=True)
                patient_class_counts = {
                    class_name_map[int(label)]: int(count)
                    for label, count in zip(unique_patients, patient_counts)
                }
                self.log_update.emit(
                    f"Patient-level class counts (all patients): {patient_class_counts}"
                )

                if self.cv_mode == "Leave-One-Subject-Out (LOSO)":
                    train_patients = patients
                    test_patients = np.array([])
                    self.n_splits = len(patients)
                    self.log_update.emit(f"LOSO active. Total folds: {self.n_splits}")
                else:
                    unique_classes, class_freq = np.unique(patient_labels, return_counts=True)
                    if len(unique_classes) < 2:
                        raise RuntimeError("Need at least two classes for patient-level stratified split.")
                    if np.min(class_freq) < 2:
                        raise RuntimeError("At least one class has fewer than 2 patients; cannot perform stratified 80/20 split.")

                    test_size = compute_stratified_split_size(
                        len(patients),
                        len(unique_classes),
                        self.test_size,
                    )
                    train_patients, test_patients = train_test_split(
                        patients,
                        test_size=test_size,
                        random_state=self.random_state,
                        shuffle=True,
                        stratify=patient_labels,
                    )

                train_patient_labels = np.array([patient_to_label[pid] for pid in train_patients])
                unique_train_labels, train_patient_counts = np.unique(train_patient_labels, return_counts=True)
                train_class_counts = {
                    class_name_map[int(label)]: int(count)
                    for label, count in zip(unique_train_labels, train_patient_counts)
                }
                self.log_update.emit(
                    f"Patient-level class counts (train split): {train_class_counts}"
                )
                
                if self.cv_mode != "Leave-One-Subject-Out (LOSO)":
                    unique_train, train_freq = np.unique(train_patient_labels, return_counts=True)
                    if self.n_splits <= 1:
                        if np.min(train_freq) < 2:
                            underfilled = {
                                class_name_map[int(label)]: int(count)
                                for label, count in zip(unique_train, train_freq)
                                if count < 2
                            }
                            raise RuntimeError(
                                "Need at least 2 patients per class in training split for stratified train/val split. "
                                f"Minimum class count is {np.min(train_freq)}. "
                                f"Classes below threshold: {underfilled}."
                            )
                    elif np.min(train_freq) < self.n_splits:
                        underfilled = {
                            class_name_map[int(label)]: int(count)
                            for label, count in zip(unique_train, train_freq)
                            if count < self.n_splits
                        }
                        raise RuntimeError(
                            f"Need at least {self.n_splits} patients per class in training split for StratifiedKFold. "
                            f"Minimum class count is {np.min(train_freq)}. "
                            f"Classes below threshold: {underfilled}."
                        )

                train_idx = np.concatenate([patient_to_indices[pid] for pid in train_patients]).astype(int)
                if len(test_patients) > 0:
                    test_idx = np.concatenate([patient_to_indices[pid] for pid in test_patients]).astype(int)
                else:
                    test_idx = np.array([], dtype=int)
                splitter = None
            elif self.multi_label:
                if len(labels_np) < 2:
                    raise RuntimeError("Need at least two samples for train/test split.")
                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                )
                train_labels = labels_np[train_idx]
                if len(train_idx) < self.n_splits:
                    raise RuntimeError(
                        f"Need at least {self.n_splits} samples for KFold cross validation."
                    )
                splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                unique_classes, class_freq = np.unique(labels_np, return_counts=True)
                if len(unique_classes) < 2:
                    raise RuntimeError("Need at least two classes for stratified train/test split.")
                if np.min(class_freq) < 2:
                    raise RuntimeError("At least one class has fewer than 2 samples; cannot perform stratified 80/20 split.")

                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=labels_np,
                )

                train_labels = labels_np[train_idx]
                unique_train, train_freq = np.unique(train_labels, return_counts=True)
                if np.min(train_freq) < self.n_splits:
                    raise RuntimeError(
                        f"Need at least {self.n_splits} samples per class in training split for StratifiedKFold. "
                        f"Minimum class count is {np.min(train_freq)}."
                    )
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            if is_patient_level:
                train_patient_count = len(np.unique(np.asarray(self.patient_ids)[train_idx]))
                test_patient_count = len(np.unique(np.asarray(self.patient_ids)[test_idx])) if len(test_idx) > 0 else 0
                self.log_update.emit(
                    f"split done (patient-level): train={len(train_idx)} samples ({train_patient_count} patients), "
                    f"test={len(test_idx)} samples ({test_patient_count} patients)"
                )
            else:
                self.log_update.emit(f"80/20 split done: train={len(train_idx)} samples, test={len(test_idx)} samples")"""

content = content[:idx_start] + new_split_logic + content[idx_end:]


# Update the split_iter assignment block
old_split_iter_start = "if self.split_by_patient:"
old_split_iter_end = "            else:\n                split_iter = splitter.split(train_idx, train_labels)"
idx_start = content.find(old_split_iter_start, idx_start + len(new_split_logic))
idx_end = content.find(old_split_iter_end, idx_start) + len(old_split_iter_end)

new_split_iter = """if is_patient_level:
                patient_ids_np = np.asarray(self.patient_ids)
                patient_to_indices = {}
                for idx, patient_id in enumerate(patient_ids_np):
                    patient_to_indices.setdefault(patient_id, []).append(idx)
                train_patients = np.unique(patient_ids_np[train_idx])
                train_patient_labels = np.array([int(labels_np[patient_ids_np == pid][0]) for pid in train_patients])

                if self.cv_mode == "Leave-One-Subject-Out (LOSO)":
                    splitter = LeaveOneGroupOut()
                    split_iter = splitter.split(train_patients, train_patient_labels, groups=train_patients)
                elif self.n_splits <= 1:
                    val_size = compute_stratified_split_size(
                        len(train_patients),
                        len(np.unique(train_patient_labels)),
                        self.test_size,
                    )
                    fold_train_patients, fold_val_patients = train_test_split(
                        train_patients,
                        test_size=val_size,
                        random_state=self.random_state,
                        shuffle=True,
                        stratify=train_patient_labels,
                    )
                    split_iter = [(fold_train_patients, fold_val_patients)]
                else:
                    splitter = StratifiedKFold(
                        n_splits=self.n_splits,
                        shuffle=True,
                        random_state=self.random_state,
                    )
                    split_iter = splitter.split(train_patients, train_patient_labels)
            else:
                split_iter = splitter.split(train_idx, train_labels)"""

content = content[:idx_start] + new_split_iter + content[idx_end:]

# Update the fold index assignment
old_fold_start = "if self.split_by_patient:"
old_fold_end = "                else:\n                    fold_train_idx = train_idx[fold_train_rel]\n                    fold_val_idx = train_idx[fold_val_rel]"
idx_start = content.find(old_fold_start, idx_start + len(new_split_iter))
idx_end = content.find(old_fold_end, idx_start) + len(old_fold_end)

new_fold = """if is_patient_level:
                    if self.n_splits <= 1 and self.cv_mode != "Leave-One-Subject-Out (LOSO)":
                        fold_train_patients = fold_train_rel
                        fold_val_patients = fold_val_rel
                    else:
                        fold_train_patients = train_patients[fold_train_rel]
                        fold_val_patients = train_patients[fold_val_rel]
                    fold_train_idx = np.concatenate(
                        [patient_to_indices[pid] for pid in fold_train_patients]
                    ).astype(int)
                    fold_val_idx = np.concatenate(
                        [patient_to_indices[pid] for pid in fold_val_patients]
                    ).astype(int)
                else:
                    fold_train_idx = train_idx[fold_train_rel]
                    fold_val_idx = train_idx[fold_val_rel]"""

content = content[:idx_start] + new_fold + content[idx_end:]

# Add check for len(test_idx) > 0 at test_loader
old_test_loader = """x_test = self.x_tensor[test_idx]
            y_test = self.y_tensor[test_idx]
            z_test = self.z_tensor[test_idx]
            label_test = self.label_tensor[test_idx]
            test_loader = DataLoader(
                TensorDataset(x_test, y_test, z_test, label_test),
                batch_size=self.batch_size,
                shuffle=False,
            )"""
new_test_loader = """if len(test_idx) > 0:
                x_test = self.x_tensor[test_idx]
                y_test = self.y_tensor[test_idx]
                z_test = self.z_tensor[test_idx]
                label_test = self.label_tensor[test_idx]
                test_loader = DataLoader(
                    TensorDataset(x_test, y_test, z_test, label_test),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
            else:
                test_loader = None
                x_test = self.x_tensor[:0]
                y_test = self.y_tensor[:0]
                z_test = self.z_tensor[:0]
                label_test = self.label_tensor[:0]"""
content = content.replace(old_test_loader, new_test_loader)

# Replace final evaluation
old_final_eval = """test_loss, test_acc = self.evaluate_loader(model, test_criterion, test_loader, device)"""
new_final_eval = """if test_loader is not None:
                test_loss, test_acc = self.evaluate_loader(model, test_criterion, test_loader, device)
            else:
                test_loss, test_acc = best_val_loss, 0.0"""
content = content.replace(old_final_eval, new_final_eval)

with open(filename, 'w') as f:
    f.write(content)

print("Patched machinelearning_testing.py")
