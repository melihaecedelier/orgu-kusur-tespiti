import os
import sys
import numpy as np
import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), '.mpl_cache')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

print(f"Project directory: {BASE_DIR}")
print(f"Figures directory: {FIG_DIR}")
print(f"TensorFlow Version: {tf.__version__}")

# ==========================================
# 1. REALISTIC KNITWEAR DATASET GENERATION
# ==========================================
def generate_knit_texture(size=128, is_defective=False, defect_type="none"):
    # Knit fabric grid
    x = np.linspace(0, 16 * np.pi, size)
    y = np.linspace(0, 16 * np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Interlocking wales and courses
    wale_freq = 0.8
    course_freq = 0.8
    knit_base = (np.sin(X * wale_freq) * np.cos(Y * course_freq) + 
                 0.4 * np.sin(2 * X * wale_freq) +
                 0.25 * np.cos(2 * Y * course_freq))
    
    # Yarn texture noise
    fiber_noise = np.random.normal(0, 0.2, (size, size))
    texture = knit_base + fiber_noise
    
    # Normalize base texture
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
    texture = 0.3 + 0.5 * texture
    
    # Introduce Defects if defective
    if is_defective:
        if defect_type == "none":
            defect_type = np.random.choice(["hole", "stain", "broken_stitch"])
            
        cx, cy = np.random.randint(30, size - 30, size=2)
        radius = np.random.randint(10, 20)
        dist_px = np.sqrt((np.arange(size)[:, None] - cy)**2 + (np.arange(size)[None, :] - cx)**2)
        
        if defect_type == "hole":
            # Broken loops / void hole
            mask = dist_px < radius
            texture[mask] = np.random.normal(0.08, 0.03, np.sum(mask))
            edge_mask = (dist_px >= radius) & (dist_px < radius + 4)
            texture[edge_mask] += np.random.uniform(0.15, 0.35, np.sum(edge_mask))
            
        elif defect_type == "stain":
            # Oil stain / discoloration
            smooth_drop = np.exp(- (dist_px**2) / (2 * (radius + 2)**2))
            texture = texture * (1.0 - 0.7 * smooth_drop)
            
        elif defect_type == "broken_stitch":
            # Snag / dropped loop streak
            length = np.random.randint(25, 50)
            y_start = max(0, cy - 3)
            y_end = min(size, cy + 3)
            x_start = max(0, cx - length // 2)
            x_end = min(size, cx + length // 2)
            texture[y_start:y_end, x_start:x_end] = np.random.uniform(0.88, 1.0, (y_end - y_start, x_end - x_start))

    return np.clip(texture, 0.0, 1.0).astype(np.float32)

print("Generating Knitwear Fabric Defect Dataset...")
NUM_NORMAL = 800
NUM_DEFECTIVE = 200 # Real-world 80:20 Imbalanced dataset

images, labels, defect_types_list = [], [], []

for i in range(NUM_NORMAL):
    images.append(generate_knit_texture(size=128, is_defective=False))
    labels.append(0)
    defect_types_list.append("Normal")

defect_kinds = ["hole", "stain", "broken_stitch"]
for i in range(NUM_DEFECTIVE):
    d_kind = defect_kinds[i % len(defect_kinds)]
    images.append(generate_knit_texture(size=128, is_defective=True, defect_type=d_kind))
    labels.append(1)
    defect_types_list.append(d_kind)

X = np.array(images)[..., np.newaxis]
y = np.array(labels, dtype=np.float32)

print(f"Total Dataset: X = {X.shape}, y = {y.shape}")

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("Generating EDA Visualizations...")

# Visual 1: Sample Images Grid
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
sample_normals = np.where(y == 0)[0][:4]
sample_defects = np.where(y == 1)[0][:4]

for i, idx in enumerate(sample_normals):
    axes[0, i].imshow(X[idx, :, :, 0], cmap='gray')
    axes[0, i].set_title(f"Hatasız Örgü (Örnek {i+1})", fontsize=11, fontweight='bold', color='navy')
    axes[0, i].axis('off')

for i, idx in enumerate(sample_defects):
    d_type_name = {"hole": "Delik/Kaçık", "stain": "Yağ Lekesi", "broken_stitch": "İlmek Düğümü"}.get(defect_types_list[idx], "Kusur")
    axes[1, i].imshow(X[idx, :, :, 0], cmap='gray')
    axes[1, i].set_title(f"Kusurlu Örgü: {d_type_name}", fontsize=11, fontweight='bold', color='darkred')
    axes[1, i].axis('off')

plt.suptitle("Görsel 1: Hatasız ve Kusurlu Örgü Kumaş Örnekleri (EDA)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "eda_1_sample_images.png")
plt.savefig(fig1_path, dpi=200)
plt.close()

# Visual 2: Class Imbalance
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["Hatasız (Normal)", "Kusurlu (Defective)"], [NUM_NORMAL, NUM_DEFECTIVE], color=['#2b5c8f', '#c0392b'], width=0.5, edgecolor='black')
ax.set_ylabel("Örnek Sayısı", fontsize=11, fontweight='bold')
ax.set_title("Görsel 2: Veri Seti Sınıf Dağılımı (Sınıf Dengesizliği)", fontsize=12, fontweight='bold')
ax.set_ylim(0, 950)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{int(yval)} (%{yval/10:.1f})", ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "eda_2_class_distribution.png")
plt.savefig(fig2_path, dpi=200)
plt.close()

# Visual 3: Pixel Intensity Distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(X[y == 0].flatten()[:100000], bins=50, alpha=0.6, label='Hatasız Örgü Pikselleri', color='#2980b9', density=True)
ax.hist(X[y == 1].flatten()[:100000], bins=50, alpha=0.6, label='Kusurlu Örgü Pikselleri', color='#e74c3c', density=True)
ax.set_title("Görsel 3: Piksel Yoğunluk Dağılımı (Normal vs Kusurlu)", fontsize=12, fontweight='bold')
ax.set_xlabel("Normalize Edilmiş Piksel Değeri [0.0 - 1.0]", fontsize=11)
ax.set_ylabel("Yoğunluk", fontsize=11)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "eda_3_pixel_intensity_hist.png")
plt.savefig(fig3_path, dpi=200)
plt.close()

# ==========================================
# 3. STRATIFIED TRAIN / VAL / TEST SPLIT
# ==========================================
indices_0 = np.where(y == 0)[0]
indices_1 = np.where(y == 1)[0]
np.random.shuffle(indices_0)
np.random.shuffle(indices_1)

n0, n1 = len(indices_0), len(indices_1)
train_idx = np.concatenate([indices_0[:int(0.70*n0)], indices_1[:int(0.70*n1)]])
val_idx   = np.concatenate([indices_0[int(0.70*n0):int(0.85*n0)], indices_1[int(0.70*n1):int(0.85*n1)]])
test_idx  = np.concatenate([indices_0[int(0.85*n0):], indices_1[int(0.85*n1):]])

np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]

# Class weights
total = len(y_train)
pos = np.sum(y_train)
neg = total - pos
class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Class Weights: {class_weights}")

# ==========================================
# 4. CNN ARCHITECTURE (Robust Defect Detector)
# ==========================================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten preserves spatial localization of defects
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="KnitDefect_CNN")
    
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()

# ==========================================
# 5. MODEL TRAINING
# ==========================================
EPOCHS = 12
BATCH_SIZE = 32

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

model_save_path = os.path.join(BASE_DIR, "knit_defect_cnn_model.keras")
model.save(model_save_path)

# ==========================================
# 6. EVALUATION & METRICS
# ==========================================
loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
f1_val = 2 * (prec * rec) / (prec + rec + 1e-7)

print(f"\n--- TEST METRICS ---")
print(f"Accuracy:  %{acc*100:.2f}")
print(f"Precision: %{prec*100:.2f}")
print(f"Recall:    %{rec*100:.2f}")
print(f"F1-Score:  {f1_val:.4f}")

# Visual 4: Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
epochs_range = range(1, len(history.history['loss']) + 1)

ax1.plot(epochs_range, history.history['loss'], 'o-', label='Eğitim Kaybı', color='#2980b9', lw=2)
ax1.plot(epochs_range, history.history['val_loss'], 's--', label='Doğrulama Kaybı', color='#e74c3c', lw=2)
ax1.set_title('Eğitim ve Doğrulama Kayıp (Loss) Grafiği', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Binary Crossentropy Kayıp', fontsize=11)
ax1.legend()

ax2.plot(epochs_range, history.history['accuracy'], 'o-', label='Eğitim Doğruluğu', color='#27ae60', lw=2)
ax2.plot(epochs_range, history.history['val_accuracy'], 's--', label='Doğrulama Doğruluğu', color='#8e44ad', lw=2)
ax2.set_title('Eğitim ve Doğrulama Doğruluk (Accuracy) Grafiği', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Doğruluk Oranı', fontsize=11)
ax2.legend()

plt.suptitle("Görsel 4: Model Eğitim Süreci Performans Eğrileri", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "eval_1_learning_curves.png")
plt.savefig(fig4_path, dpi=200, bbox_inches='tight')
plt.close()

# Visual 5: Confusion Matrix
y_pred_probs = model.predict(X_test).ravel()
y_pred_binary = (y_pred_probs >= 0.5).astype(int)
y_true_binary = y_test.astype(int)

TN = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
FP = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
FN = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
TP = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
cm = np.array([[TN, FP], [FN, TP]])

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        label = f"{val}\n" + (
            "(TN - Doğru Negatif)" if i==0 and j==0 else
            "(FP - Yanlış Alarm)" if i==0 and j==1 else
            "(FN - Kaçırılan Kusur)" if i==1 and j==0 else
            "(TP - Yakalanan Kusur)"
        )
        color = "white" if val > (cm.max()/2) else "black"
        ax.text(j, i, label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)

fig.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Tahmin: Normal', 'Tahmin: Kusurlu'], fontsize=11, fontweight='bold')
ax.set_yticklabels(['Gerçek: Normal', 'Gerçek: Kusurlu'], fontsize=11, fontweight='bold', rotation=90, va='center')
ax.set_title("Görsel 5: Test Verisi Karışıklık Matrisi (Confusion Matrix)", fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "eval_2_confusion_matrix.png")
plt.savefig(fig5_path, dpi=200)
plt.close()

# Visual 6: Prediction Inspection / Error Analysis
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Pick sample indices
tp_indices = np.where((y_true_binary == 1) & (y_pred_binary == 1))[0]
tn_indices = np.where((y_true_binary == 0) & (y_pred_binary == 0))[0]
fp_indices = np.where((y_true_binary == 0) & (y_pred_binary == 1))[0]
fn_indices = np.where((y_true_binary == 1) & (y_pred_binary == 0))[0]

# Subplot 1: TP
if len(tp_indices) > 0:
    idx = tp_indices[0]
    axes[0, 0].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[0, 0].set_title(f"Doğru Tespit: Kusurlu (TP)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Kusurlu", color='green', fontweight='bold')
axes[0, 0].axis('off')

# Subplot 2: TN
if len(tn_indices) > 0:
    idx = tn_indices[0]
    axes[0, 1].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[0, 1].set_title(f"Doğru Tespit: Hatasız (TN)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Normal", color='navy', fontweight='bold')
axes[0, 1].axis('off')

# Subplot 3: FP or second TN
if len(fp_indices) > 0:
    idx = fp_indices[0]
    axes[1, 0].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[1, 0].set_title(f"HATA: Yanlış Alarm (FP)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Normal", color='red', fontweight='bold')
elif len(tn_indices) > 1:
    idx = tn_indices[1]
    axes[1, 0].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[1, 0].set_title(f"Doğru Tespit (İkinci TN)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Normal", color='navy', fontweight='bold')
axes[1, 0].axis('off')

# Subplot 4: FN or second TP
if len(fn_indices) > 0:
    idx = fn_indices[0]
    axes[1, 1].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[1, 1].set_title(f"KRİTİK HATA: Kaçan Kusur (FN)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Kusurlu", color='darkred', fontweight='bold')
elif len(tp_indices) > 1:
    idx = tp_indices[1]
    axes[1, 1].imshow(X_test[idx, :, :, 0], cmap='gray')
    axes[1, 1].set_title(f"Kusursuz Tespit (İkinci TP)\nOlasılık: %{y_pred_probs[idx]*100:.1f} | Gerçek: Kusurlu", color='green', fontweight='bold')
axes[1, 1].axis('off')

plt.suptitle("Görsel 6: Model Tahmin ve Hata İncelemesi", fontsize=13, fontweight='bold')
plt.tight_layout()
fig6_path = os.path.join(FIG_DIR, "eval_3_error_analysis.png")
plt.savefig(fig6_path, dpi=200)
plt.close()

# Save metrics text
summary = f"""EVALUATION METRICS SUMMARY:
==================================
Accuracy:  %{acc*100:.2f}
Precision: %{prec*100:.2f}
Recall:    %{rec*100:.2f}
F1-Score:  {f1_val:.4f}
Loss:      {loss:.4f}

Confusion Matrix:
TN: {TN}, FP: {FP}
FN: {FN}, TP: {TP}
"""

with open(os.path.join(BASE_DIR, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print("Pipeline Successfully Completed!")
