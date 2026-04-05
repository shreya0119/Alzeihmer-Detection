"""
ENHANCED Alzheimer Detection - Training Analysis
Professional visualization with improved formatting
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class TrainingAnalyzer:
    def __init__(self, history_file='training_history.json'):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self):
        if not Path(self.history_file).exists():
            raise FileNotFoundError(f"❌ {self.history_file} not found!")

        with open(self.history_file, 'r') as f:
            return json.load(f)

    def plot_graphs_enhanced(self, save_path='training_analysis_enhanced.png'):
        """Enhanced 2x2 visualization with professional formatting"""

        # ===== SETUP =====
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0a0e27')  # Dark blue background

        # Overall title
        fig.suptitle(
            'MobileNetV2 Alzheimer Detection - Training Analysis',
            fontsize=18,
            fontweight='bold',
            color='#00d4ff',
            y=0.98
        )

        epochs = range(1, len(self.history['loss']) + 1)

        # Color scheme
        color_train = '#00d4ff'      # Cyan
        color_val = '#ff6b35'        # Orange

        # ===== TOP LEFT: LOSS CURVE =====
        ax = axes[0, 0]
        ax.plot(epochs, self.history['loss'],
               color=color_train, linewidth=3, marker='o', markersize=6,
               label='Training Loss', alpha=0.9)
        ax.plot(epochs, self.history['val_loss'],
               color=color_val, linewidth=3, marker='s', markersize=6,
               label='Validation Loss', alpha=0.9)

        ax.set_title('Training vs Validation Loss',
                    fontsize=14, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Epoch', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, color='white', fontweight='bold')
        ax.legend(loc='upper right', fontsize=11,
                 facecolor='#1a2550', edgecolor='#00d4ff', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#0f1535')
        ax.tick_params(colors='white', labelsize=10)

        # Add value annotations at start and end
        ax.text(1, self.history['loss'][0], f"{self.history['loss'][0]:.3f}",
               fontsize=9, color=color_train, fontweight='bold', va='bottom')
        ax.text(len(epochs), self.history['loss'][-1], f"{self.history['loss'][-1]:.3f}",
               fontsize=9, color=color_train, fontweight='bold', ha='right', va='bottom')

        # ===== TOP RIGHT: LOSS CURVE (ZOOMED) =====
        ax = axes[0, 1]
        ax.plot(epochs, self.history['loss'],
               color=color_train, linewidth=3, marker='o', markersize=6,
               label='Training Loss', alpha=0.9)
        ax.plot(epochs, self.history['val_loss'],
               color=color_val, linewidth=3, marker='s', markersize=6,
               label='Validation Loss', alpha=0.9)

        # Focus on last half for detail
        if len(epochs) > 5:
            ax.set_xlim(len(epochs)//2, len(epochs))

        ax.set_title('Loss Curve - Zoomed View (Later Epochs)',
                    fontsize=14, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Epoch', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, color='white', fontweight='bold')
        ax.legend(loc='best', fontsize=11,
                 facecolor='#1a2550', edgecolor='#00d4ff', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#0f1535')
        ax.tick_params(colors='white', labelsize=10)

        # ===== BOTTOM LEFT: ACCURACY CURVE =====
        ax = axes[1, 0]
        ax.plot(epochs, self.history['accuracy'],
               color=color_train, linewidth=3, marker='o', markersize=6,
               label='Training Accuracy', alpha=0.9)
        ax.plot(epochs, self.history['val_accuracy'],
               color=color_val, linewidth=3, marker='s', markersize=6,
               label='Validation Accuracy', alpha=0.9)

        ax.set_title('Training vs Validation Accuracy',
                    fontsize=14, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Epoch', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, color='white', fontweight='bold')
        ax.set_ylim([min(self.history['accuracy'] + self.history['val_accuracy']) - 0.05, 1.0])
        ax.legend(loc='lower right', fontsize=11,
                 facecolor='#1a2550', edgecolor='#00d4ff', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#0f1535')
        ax.tick_params(colors='white', labelsize=10)

        # Add value annotations
        ax.text(1, self.history['accuracy'][0], f"{self.history['accuracy'][0]:.3f}",
               fontsize=9, color=color_train, fontweight='bold', va='bottom')
        ax.text(len(epochs), self.history['val_accuracy'][-1],
               f"{self.history['val_accuracy'][-1]:.3f}",
               fontsize=9, color=color_val, fontweight='bold', ha='right', va='bottom')

        # ===== BOTTOM RIGHT: SUMMARY STATISTICS TABLE =====
        ax = axes[1, 1]
        ax.axis('off')  # Hide axes for table

        # Calculate statistics
        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        train_acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']

        best_val_acc_idx = np.argmax(val_acc)
        best_val_acc = val_acc[best_val_acc_idx]

        overfitting_gap = abs(train_acc[-1] - val_acc[-1])
        loss_reduction = ((train_loss[0] - train_loss[-1]) / train_loss[0]) * 100

        # Create table data
        table_data = [
            ['METRIC', 'INITIAL', 'FINAL', 'CHANGE'],
            ['', '', '', ''],
            ['Training Loss', f'{train_loss[0]:.4f}', f'{train_loss[-1]:.4f}',
             f'↓ {loss_reduction:.1f}%'],
            ['Validation Loss', f'{val_loss[0]:.4f}', f'{val_loss[-1]:.4f}',
             f'↓ {((val_loss[0]-val_loss[-1])/val_loss[0]*100):.1f}%'],
            ['', '', '', ''],
            ['Training Accuracy', f'{train_acc[0]*100:.2f}%', f'{train_acc[-1]*100:.2f}%',
             f'↑ {(train_acc[-1]-train_acc[0])*100:.2f}%'],
            ['Validation Accuracy', f'{val_acc[0]*100:.2f}%', f'{val_acc[-1]*100:.2f}%',
             f'↑ {(val_acc[-1]-val_acc[0])*100:.2f}%'],
            ['', '', '', ''],
            ['Best Validation Acc', f'{best_val_acc*100:.2f}%', f'@ Epoch {best_val_acc_idx+1}', ''],
            ['Overfitting Gap', f'{overfitting_gap*100:.2f}%', '', ''],
        ]

        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#00d4ff')
            cell.set_text_props(weight='bold', color='black', fontsize=11)

        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#1a2550')
                else:
                    cell.set_facecolor('#0f1535')
                cell.set_text_props(color='white', fontsize=10)

                # Highlight important metrics
                if i in [2, 3, 6, 7, 9]:
                    cell.set_text_props(fontweight='bold')

        # Status indicator
        if overfitting_gap < 0.05:
            status = "✅ EXCELLENT"
            status_color = '#00ff00'
        elif overfitting_gap < 0.10:
            status = "✅ GOOD"
            status_color = '#00ff00'
        elif overfitting_gap < 0.15:
            status = "⚠️ MODERATE"
            status_color = '#ffff00'
        else:
            status = "❌ SEVERE"
            status_color = '#ff0000'

        ax.text(0.5, -0.15, f"Model Status: {status}",
               ha='center', va='top', fontsize=13, fontweight='bold',
               color=status_color, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#0a0e27', edgecolor=status_color, linewidth=2))

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=300, facecolor='#0a0e27', bbox_inches='tight',
                   edgecolor='#00d4ff', pad_inches=0.3)
        print(f"\n✅ Enhanced graph saved: {save_path}")
        plt.show()

    def print_report(self):
        """Print detailed statistics"""

        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        train_acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']

        print("\n" + "="*80)
        print("🧠 ALZHEIMER DETECTION - MOBILENETV2 TRAINING ANALYSIS REPORT".center(80))
        print("="*80)

        print("\n📊 LOSS METRICS:")
        print(f"  {'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
        print(f"  {'-'*70}")
        loss_reduction = ((train_loss[0] - train_loss[-1]) / train_loss[0]) * 100
        print(f"  {'Training Loss':<25} {train_loss[0]:<15.4f} {train_loss[-1]:<15.4f} ↓ {loss_reduction:.1f}%")
        val_loss_reduction = ((val_loss[0] - val_loss[-1]) / val_loss[0]) * 100
        print(f"  {'Validation Loss':<25} {val_loss[0]:<15.4f} {val_loss[-1]:<15.4f} ↓ {val_loss_reduction:.1f}%")

        print("\n📈 ACCURACY METRICS:")
        print(f"  {'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
        print(f"  {'-'*70}")
        print(f"  {'Training Accuracy':<25} {train_acc[0]*100:<14.2f}% {train_acc[-1]*100:<14.2f}% ↑ {(train_acc[-1]-train_acc[0])*100:>6.2f}%")
        print(f"  {'Validation Accuracy':<25} {val_acc[0]*100:<14.2f}% {val_acc[-1]*100:<14.2f}% ↑ {(val_acc[-1]-val_acc[0])*100:>6.2f}%")

        print("\n⚖️ OVERFITTING ANALYSIS:")
        gap = abs(train_acc[-1] - val_acc[-1])
        print(f"  Final Accuracy Gap: {gap*100:.2f}%")

        if gap < 0.05:
            status = "✅ EXCELLENT - Model is well-balanced"
        elif gap < 0.10:
            status = "✅ GOOD - Acceptable overfitting levels"
        elif gap < 0.15:
            status = "⚠️ MODERATE - Some overfitting present"
        else:
            status = "❌ SEVERE - Significant overfitting detected"

        print(f"  Status: {status}")

        print("\n🎯 CONVERGENCE METRICS:")
        best_epoch = np.argmax(val_acc)
        print(f"  Best Validation Accuracy: {val_acc[best_epoch]*100:.2f}% @ Epoch {best_epoch + 1}/{len(train_loss)}")
        print(f"  Training Stability: {'Stable' if np.std(np.diff(val_acc[-3:])) < 0.01 else 'Unstable'}")

        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("🧠 Loading training analysis...")
    analyzer = TrainingAnalyzer('training_history.json')

    print("📊 Generating enhanced visualization...")
    analyzer.plot_graphs_enhanced()

    print("📋 Generating detailed report...")
    analyzer.print_report()

    print("✅ Analysis complete!")