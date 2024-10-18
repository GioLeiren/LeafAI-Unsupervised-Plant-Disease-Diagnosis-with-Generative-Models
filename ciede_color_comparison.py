import os
import cv2
import numpy as np
from colorspacious import cspace_convert
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_ciede2000(image1_path, image2_path):
    # Carregue as imagens
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1.shape != img2.shape:
        raise ValueError("Images must be the same size")

    # Converta as imagens para o espaço de cores LAB
    img1_lab = cspace_convert(img1, "sRGB1", "CIELab")
    img2_lab = cspace_convert(img2, "sRGB1", "CIELab")

    # Inicialize a diferença de cor
    differences = np.linalg.norm(img1_lab - img2_lab, axis=2)
    mean_color_difference = np.mean(differences)

    return mean_color_difference, differences

def create_heatmap(differences, output_path):
    # Normalize as diferenças para o intervalo [0, 255]
    normalized_differences = cv2.normalize(differences, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized_differences.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)

def calculate_metrics(ground_truth_folder, generated_folder, output_folder, threshold):
    y_true = []
    y_pred = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ground_truth_paths = sorted([os.path.join(ground_truth_folder, f) for f in os.listdir(ground_truth_folder) if f.endswith('real_B.png')])
    generated_paths = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) if f.endswith('fake_B.png')])

    for gt_path, gen_path in zip(ground_truth_paths, generated_paths):
        base_filename = os.path.basename(gt_path).replace('real_B', 'heatmap')
        output_path = os.path.join(output_folder, base_filename)
        
        mean_diff, differences = calculate_ciede2000(gt_path, gen_path)
        create_heatmap(differences, output_path)
        
        for i in range(differences.shape[0]):
            for j in range(differences.shape[1]):
                y_true.append(0 if 'healthy' in gt_path else 1)  # 0 para saudável, 1 para doente
                y_pred.append(1 if differences[i, j] > threshold else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Verifique ambas as classes em y_true antes de calcular AUC
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = float('nan')  # AUC não é definida nesse caso

    return precision, recall, f1, auc

# Parâmetros e pastas
ground_truth_folder = './results/Leafs_pix2pix/test_latest/images'
generated_folder = './results/Leafs_pix2pix/test_latest/images'
output_folder = './results/Leafs_pix2pix/test_latest/images/heatmaps'
threshold = 10.0

precision, recall, f1, auc = calculate_metrics(ground_truth_folder, generated_folder, output_folder, threshold)
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')
