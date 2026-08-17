"""
MT-LNN 3D交互式可视化
展示13根protofilaments和5个时间尺度的液态神经网络结构

依赖: plotly
安装: pip install plotly
运行: python plot_mt_lnn_3d_interactive.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MT-LNN配置
N_PROTOS = 13  # 生物学常数
N_SCALES = 5   # 时间尺度数量
TAU_MIN, TAU_MAX = 0.01, 10.0

# 生成时间尺度（几何分布）
time_scales = np.geomspace(TAU_MIN, TAU_MAX, N_SCALES)
print(f"Time scales (τ): {time_scales}")

# ============================================================================
# 图1：13根Protofilaments的3D柱状图
# ============================================================================

# 为每个proto生成模拟的激活强度
np.random.seed(42)
activation_matrix = np.random.rand(N_PROTOS, N_SCALES) * 10

# 圆形排列13根protofilaments
angles = np.linspace(0, 2 * np.pi, N_PROTOS, endpoint=False)
radius = 5

# 创建子图
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "13 Protofilaments × 5 Time Scales",
        "Lateral Coupling Network",
        "GTP Hydrolysis Dynamics",
        "Resonance Activation Heatmap"
    ),
    specs=[
        [{'type': 'surface'}, {'type': 'scatter3d'}],
        [{'type': 'scatter'}, {'type': 'heatmap'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# ============================================================================
# 子图1：Protofilament激活的3D表面
# ============================================================================

# 创建网格
proto_indices = np.arange(N_PROTOS)
scale_indices = np.arange(N_SCALES)
X_grid, Y_grid = np.meshgrid(proto_indices, scale_indices)

fig.add_trace(
    go.Surface(
        x=X_grid,
        y=Y_grid,
        z=activation_matrix.T,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(x=0.46, len=0.4, title="Activation"),
        name="Resonance",
        hovertemplate='Proto: %{x}<br>Scale: %{y}<br>Act: %{z:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# ============================================================================
# 子图2：横向耦合的3D网络图
# ============================================================================

# 生成protofilaments的3D位置（圆柱形排列）
x_coords = radius * np.cos(angles)
y_coords = radius * np.sin(angles)
z_coords = np.zeros(N_PROTOS)  # 基准高度

# 添加protofilaments节点
fig.add_trace(
    go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+text',
        marker=dict(size=10, color=np.arange(N_PROTOS), colorscale='Rainbow', showscale=False),
        text=[f"P{i}" for i in range(N_PROTOS)],
        textposition="top center",
        name="Protofilaments",
        hovertemplate='Protofilament %{text}<extra></extra>'
    ),
    row=1, col=2
)

# 添加横向耦合连接（B晶格键）
edge_x, edge_y, edge_z = [], [], []
for i in range(N_PROTOS):
    j = (i + 1) % N_PROTOS  # 环形连接
    edge_x.extend([x_coords[i], x_coords[j], None])
    edge_y.extend([y_coords[i], y_coords[j], None])
    edge_z.extend([z_coords[i], z_coords[j], None])

fig.add_trace(
    go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='rgba(100, 100, 100, 0.5)', width=3),
        name="Lateral Coupling",
        showlegend=False,
        hoverinfo='skip'
    ),
    row=1, col=2
)

# ============================================================================
# 子图3：GTP水解动力学曲线
# ============================================================================

# 模拟GTP衰减曲线
t = np.linspace(0, 256, 100)  # gtp_period = 256
gamma = 0.1
gtp_decay = np.exp(-gamma * t / 256)

fig.add_trace(
    go.Scatter(
        x=t,
        y=gtp_decay,
        mode='lines',
        line=dict(color='#42949E', width=3),
        name="GTP Cap",
        fill='tozeroy',
        fillcolor='rgba(66, 148, 158, 0.3)',
        hovertemplate='Token: %{x:.0f}<br>GTP: %{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)

# 标记更新周期
for period in [0, 256]:
    fig.add_vline(x=period, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text=f"Renewal @ {period}")

# ============================================================================
# 子图4：共振激活热力图
# ============================================================================

fig.add_trace(
    go.Heatmap(
        z=activation_matrix,
        x=[f"τ{i+1}" for i in range(N_SCALES)],
        y=[f"P{i}" for i in range(N_PROTOS)],
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(x=1.1, len=0.4),
        hovertemplate='Proto: %{y}<br>Scale: %{x}<br>Act: %{z:.2f}<extra></extra>'
    ),
    row=2, col=2
)

# ============================================================================
# 布局优化
# ============================================================================

fig.update_layout(
    title={
        'text': "MT-LNN Architecture 3D Visualization<br><sub>Microtubule-Inspired Liquid Neural Network</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    height=900,
    showlegend=True,
    template='plotly_white',
    font=dict(family="Arial, sans-serif", size=11),
    margin=dict(t=120, b=50, l=50, r=150)
)

# 子图1坐标轴
fig.update_scenes(
    dict(
        xaxis_title="Protofilament",
        yaxis_title="Time Scale",
        zaxis_title="Activation",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    row=1, col=1
)

# 子图2坐标轴
fig.update_scenes(
    dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=0, y=0, z=2.5)),
        aspectmode='cube'
    ),
    row=1, col=2
)

# 子图3坐标轴
fig.update_xaxes(title_text="Token Position", row=2, col=1)
fig.update_yaxes(title_text="GTP Level", range=[0, 1.1], row=2, col=1)

# 子图4坐标轴
fig.update_xaxes(title_text="Time Scale", row=2, col=2)
fig.update_yaxes(title_text="Protofilament", row=2, col=2)

# ============================================================================
# 保存和显示
# ============================================================================

# 保存为HTML（完全交互式）
output_file = "mt_lnn_3d_interactive.html"
fig.write_html(output_file)
print(f"✅ Saved interactive HTML: {output_file}")

# 保存为静态图片
fig.write_image("mt_lnn_3d_interactive.png", width=1600, height=900, scale=2)
print(f"✅ Saved static PNG: mt_lnn_3d_interactive.png")

# 在浏览器中打开
fig.show()

print("\n" + "="*60)
print("MT-LNN 3D Visualization Summary")
print("="*60)
print(f"Protofilaments: {N_PROTOS} (biological constant)")
print(f"Time Scales: {N_SCALES} (geometric: {TAU_MIN} → {TAU_MAX})")
print(f"Total LTC channels: {N_PROTOS * N_SCALES} = {N_PROTOS}×{N_SCALES}")
print("\nVisualization Components:")
print("  1. Surface Plot: 3D activation landscape")
print("  2. Network Graph: Lateral coupling topology")
print("  3. Time Series: GTP hydrolysis dynamics")
print("  4. Heatmap: Per-channel resonance intensity")
print("="*60)
