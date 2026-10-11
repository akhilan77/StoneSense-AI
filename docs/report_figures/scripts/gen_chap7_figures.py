"""Script to generate Chapter 7 Explainability figures:
- fig_7_1_global_shap_summary.png: Global SHAP summary beeswarm plot
- fig_7_2_local_shap_waterfall.png: Local SHAP waterfall for single sample
- fig_7_3_gradcam_resnet18_side_by_side.png: 4 classes, source CT beside Grad-CAM overlay

Outputs go in docs/report_figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig_7_1():
    """Figure 7.1: Global SHAP summary plot."""
    src = Path("ml/outputs/shap/summary_plot.png")
    dst = OUTPUT_DIR / "fig_7_1_global_shap_summary.png"
    
    im = Image.open(src)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor('white')
    ax.imshow(im)
    ax.axis('off')
    
    fig.suptitle("Figure 7.1: Global SHAP Beeswarm Summary Plot (Urine Risk Features)",
                 fontsize=12, fontweight='bold', y=0.97, color='#0f172a')
    
    fig.text(0.5, 0.02,
             "SHAP feature attributions on urine chemistry features across all patient observations.\n"
             "High feature values are plotted in red, low values in blue. Source: ml/outputs/shap/summary_plot.png",
             ha='center', fontsize=8.5, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(dst, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {dst}")


def generate_fig_7_2():
    """Figure 7.2: Local SHAP waterfall for one real sample row, no identifiers."""
    src = Path("ml/outputs/shap/waterfall_patient_01.png")
    dst = OUTPUT_DIR / "fig_7_2_local_shap_waterfall.png"
    
    im = Image.open(src)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.imshow(im)
    ax.axis('off')
    
    fig.suptitle("Figure 7.2: Local Patient Risk Attribution Waterfall (De-Identified Clinical Sample)",
                 fontsize=12, fontweight='bold', y=0.97, color='#0f172a')
    
    fig.text(0.5, 0.02,
             "Waterfall decomposition of physiological biomarker contributions driving individual stone risk prediction: f(x) vs E[f(x)].\n"
             "De-identified observation from clinical urine cohort. Source: ml/outputs/shap/waterfall_patient_01.png",
             ha='center', fontsize=8.5, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(dst, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {dst}")


def generate_fig_7_3():
    """Figure 7.3: Grad-CAM, ResNet18: for each class, source CT beside its overlay."""
    pairs = [
        ("Normal Renal Parenchyma", "Normal- (1174)",
         "dl/processed_grouped/train/Normal/Normal- (1174).jpg",
         "dl/outputs/gradcam/Normal/Normal- (1174)_gradcam.png",
         "Diffuse, non-focal cortical attention"),
        ("Renal Cortical Cyst", "Cyst- (1158)",
         "dl/processed_grouped/validation/Cyst/Cyst- (1158).jpg",
         "dl/outputs/gradcam/Cyst/Cyst- (1158)_gradcam.png",
         "Attention localized to hypodense cyst margin"),
        ("Solid Renal Tumor", "Tumor- (1107)",
         "dl/processed_grouped/train/Tumor/Tumor- (1107).jpg",
         "dl/outputs/gradcam/Tumor/Tumor- (1107)_gradcam.png",
         "Strong focal activation over exophytic tumor mass"),
        ("Renal Calculus (Stone)", "Stone- (1012)",
         "dl/processed_grouped/train/Stone/Stone- (1012).jpg",
         "dl/outputs/gradcam/Stone/Stone- (1012)_gradcam.png",
         "Dense spatial focus centered on hyperdense calculus")
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 13), dpi=300)
    fig.patch.set_facecolor('white')

    for i, (cname, fname, orig_p, grad_p, desc) in enumerate(pairs):
        im_orig = Image.open(orig_p).convert('RGB')
        im_grad = Image.open(grad_p).convert('RGB')
        
        ax_orig = axes[i, 0]
        ax_grad = axes[i, 1]
        
        ax_orig.imshow(im_orig)
        ax_orig.axis('off')
        ax_orig.set_title(f"{cname} — Original CT Slice\n({fname}.jpg)", fontsize=9.5, fontweight='bold', pad=6, color='#0f172a')
        
        ax_grad.imshow(im_grad)
        ax_grad.axis('off')
        ax_grad.set_title(f"ResNet18 Layer4 Grad-CAM Overlay\n({desc})", fontsize=9.5, fontweight='bold', pad=6, color='#1e3a8a')

    fig.suptitle("Figure 7.3: ResNet18 Grad-CAM Visual Explanations Across Kidney Pathology Classes",
                 fontsize=13, fontweight='bold', y=0.99, color='#0f172a')

    fig.text(0.5, 0.01,
             "Left: Unprocessed 512×512 axial CT input slice. Right: ResNet18 Layer4 Grad-CAM heat activation overlay.\n"
             "Sources: dl/processed_grouped/ and dl/outputs/gradcam/ (Generated with PyTorch Grad-CAM on Grouped Model)",
             ha='center', fontsize=8.5, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    dst = OUTPUT_DIR / "fig_7_3_gradcam_resnet18_side_by_side.png"
    plt.savefig(dst, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {dst}")


if __name__ == "__main__":
    generate_fig_7_1()
    generate_fig_7_2()
    generate_fig_7_3()
