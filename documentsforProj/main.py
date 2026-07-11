import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 1. OS별 한글 폰트 자동 설정 (한글 깨짐 현상 예방)
system_os = platform.system()
if system_os == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_os == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    # 시스템에 설치된 나눔글꼴이나 노토산스 CJK 글꼴을 우선 반영합니다.
    plt.rcParams['font.family'] = 'Noto Sans CJK KR'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 2. 데이터 파일 로딩 (제공된 고유 파일명 반영)
file_normal = 'autobench_20260528_010219.csv'
file_stress = 'autobench_20260528_014356.csv'

try:
    df_normal_raw = pd.read_csv(file_normal)
    df_stress_raw = pd.read_csv(file_stress)
except FileNotFoundError as e:
    print(f"Error: 파일을 찾을 수 없습니다. 경로와 파일명을 다시 확인해주세요. ({e})")
    exit()

# 3. 데이터 전처리 (0~49행까지의 개별 실행 데이터만 슬라이싱)
# (50행 이후는 요약 테이블 및 메타 정보이므로 집계에서 제외합니다)
df_normal = df_normal_raw.iloc[0:50].copy()
df_stress = df_stress_raw.iloc[0:50].copy()

# 데이터 타입 변환 (문자열로 읽힌 열들을 실수형으로 변환)
numeric_cols = ['avg_fps', 'avg_ft_ms', 'avg_dc', 'avg_culled']
for col in numeric_cols:
    df_normal[col] = pd.to_numeric(df_normal[col], errors='coerce')
    df_stress[col] = pd.to_numeric(df_stress[col], errors='coerce')

# 4. 알고리즘명 보기 쉽게 매핑
label_map = {
    '0.Baseline (all OFF)': 'Baseline',
    '1.Frustum Culling': 'Frustum Culling',
    '2.LOD': 'LOD',
    '3.GPU Instancing': 'GPU Instancing',
    '4.Backface Culling': 'Backface Culling',
    '5.Depth Sort': 'Depth Sort',
    '6.Occlusion Culling': 'Occlusion Culling',
    '7.View Dist Cull': 'View Distance Cull',
    '8.Small Obj Cull': 'Small Object Cull',
    '9.Deferred Shading': 'Deferred Shading'
}

df_normal['clean_label'] = df_normal['experiment'].map(label_map)
df_stress['clean_label'] = df_stress['experiment'].map(label_map)

# 5. 최적화 기법별 평균 성능 데이터 집계 (5회 반복 실험의 평균)
summary_normal = df_normal.groupby('clean_label')[numeric_cols].mean().reset_index()
summary_stress = df_stress.groupby('clean_label')[numeric_cols].mean().reset_index()

# Baseline을 기준으로 한 정렬 보장
summary_normal = summary_normal.set_index('clean_label').reindex(label_map.values()).reset_index()
summary_stress = summary_stress.set_index('clean_label').reindex(label_map.values()).reset_index()

# Baseline 대비 FPS 개선율(%) 산출
base_fps_normal = summary_normal.loc[summary_normal['clean_label'] == 'Baseline', 'avg_fps'].values[0]
base_fps_stress = summary_stress.loc[summary_stress['clean_label'] == 'Baseline', 'avg_fps'].values[0]

summary_normal['improvement_pct'] = ((summary_normal['avg_fps'] - base_fps_normal) / base_fps_normal) * 100
summary_stress['improvement_pct'] = ((summary_stress['avg_fps'] - base_fps_stress) / base_fps_stress) * 100


# ==========================================
# [시각화 1] 일반 vs 고부하 환경의 절대 FPS 수치 비교
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# 일반 부하 환경 바 차트 (오브젝트 330개)
sns.barplot(data=summary_normal, x='avg_fps', y='clean_label', ax=ax1, palette='Blues_r')
ax1.set_title('일반 환경 (오브젝트 330개) - 평균 FPS', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('평균 FPS (Frames Per Second)', fontsize=11)
ax1.set_ylabel('적용된 최적화 알고리즘', fontsize=11)
for i, v in enumerate(summary_normal['avg_fps']):
    ax1.text(v + 30, i, f"{v:.1f}", va='center', fontsize=9, fontweight='semibold')

# 고부하 환경 바 차트 (오브젝트 5,280개)
sns.barplot(data=summary_stress, x='avg_fps', y='clean_label', ax=ax2, palette='Oranges_r')
ax2.set_title('고부하 환경 (오브젝트 5,280개) - 평균 FPS', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('평균 FPS (Frames Per Second)', fontsize=11)
ax2.set_ylabel('', fontsize=11)
for i, v in enumerate(summary_stress['avg_fps']):
    ax2.text(v + 10, i, f"{v:.1f}", va='center', fontsize=9, fontweight='semibold')

plt.tight_layout()
plt.savefig('fps_absolute_comparison.png', dpi=300)
print("성공: 'fps_absolute_comparison.png' 저장 완료.")
plt.close()


# ==========================================
# [시각화 2] Baseline 대비 평균 FPS 개선율 (%) 비교 그룹 차트
# ==========================================
m_df1 = summary_normal[['clean_label', 'improvement_pct']].copy()
m_df1['환경'] = '일반 환경 (오브젝트 330개)'
m_df2 = summary_stress[['clean_label', 'improvement_pct']].copy()
m_df2['환경'] = '고부하 환경 (오브젝트 5,280개)'

combined_df = pd.concat([m_df1, m_df2], axis=0)

fig2, ax = plt.subplots(figsize=(13, 7.5))
sns.barplot(data=combined_df, x='clean_label', y='improvement_pct', hue='환경', ax=ax, palette=['#3498db', '#e67e22'])
ax.set_title('Baseline 대비 평균 FPS 개선율 (%) 비교', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('최적화 알고리즘', fontsize=12, labelpad=10)
ax.set_ylabel('FPS 개선율 (%)', fontsize=12)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
plt.xticks(rotation=30, ha='right', fontsize=10)

# 막대 상단에 수치 라벨 추가 (의미 있는 수치만 마킹)
for p in ax.patches:
    height = p.get_height()
    if abs(height) > 0.5:  # 개선율 폭이 0.5% 초과인 경우만 텍스트 노출
        ax.annotate(f"{height:+.1f}%",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fps_improvement_comparison.png', dpi=300)
print("성공: 'fps_improvement_comparison.png' 저장 완료.")
plt.close()


# ==========================================
# [시각화 3] 드로우 콜 차단 수치와 컬링 객체 수 검증 차트
# ==========================================
fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(15, 6.5))

# 드로우 콜 감소 효율 비교
sns.barplot(data=summary_stress, x='avg_dc', y='clean_label', ax=ax3_1, palette='Purples_r')
ax3_1.set_title('고부하 환경 - 평균 드로우 콜 (Draw Calls)', fontsize=13, fontweight='bold', pad=15)
ax3_1.set_xlabel('실제 렌더링된 드로우 콜 수', fontsize=11)
ax3_1.set_ylabel('최적화 알고리즘', fontsize=11)
for i, v in enumerate(summary_stress['avg_dc']):
    ax3_1.text(v + 50, i, f"{int(v):,}", va='center', fontsize=9, fontweight='semibold')

# 실제 소프트웨어 상에서 Culling(제거)된 객체 수 비교
sns.barplot(data=summary_stress, x='avg_culled', y='clean_label', ax=ax3_2, palette='Greens_r')
ax3_2.set_title('고부하 환경 - 평균 컬링된 오브젝트 수 (Culled)', fontsize=13, fontweight='bold', pad=15)
ax3_2.set_xlabel('필터링을 통해 제거된 오브젝트 수', fontsize=11)
ax3_2.set_ylabel('', fontsize=11)
for i, v in enumerate(summary_stress['avg_culled']):
    ax3_2.text(v + 50, i, f"{int(v):,}", va='center', fontsize=9, fontweight='semibold')

plt.tight_layout()
plt.savefig('culling_efficiency_comparison.png', dpi=300)
print("성공: 'culling_efficiency_comparison.png' 저장 완료.")
plt.close()

print("\n--- 모든 벤치마크 데이터 시각화 이미지 저장이 완료되었습니다! ---")