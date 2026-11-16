"""Tạo hình minh họa SVM. Chạy: uv run python docs/images/scientific/generate.py."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent
BLUE, RED, INK, GRAY = '#0F4D92', '#B64342', '#272727', '#767676'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13,
                     'axes.spines.right':False, 'axes.spines.top':False,
                     'axes.linewidth':1.6, 'legend.frameon':False,
                     'svg.fonttype':'none'})


def save(fig, name):
    fig.tight_layout(pad=2)
    for suffix in ('png', 'svg'):
        path = OUT / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        if suffix == 'svg':  # matplotlib leaves trailing spaces on path lines; strip for `git diff --check`.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def plot_points(ax, neg, pos):
    ax.scatter(*neg.T, c=BLUE, s=55, label='Lớp −1', zorder=4)
    ax.scatter(*pos.T, c=RED, marker='^', s=65, label='Lớp +1', zorder=4)
    ax.set(xlim=(-2.6,2.6), ylim=(-2.5,2.5), xlabel='$x_1$', ylabel='$x_2$')
    ax.set_aspect('equal')


def boundary(ax):
    x = np.linspace(-3,3,200)
    ax.plot(x, -x, color=INK, lw=2, label='$f(x)=0$')
    ax.plot(x, 1-x, color=GRAY, lw=1.5, ls='--', label='$f(x)=\\pm1$')
    ax.plot(x, -1-x, color=GRAY, lw=1.5, ls='--')


negative = np.array([[-2,-1.4],[-1.8,-.4],[-1.1,-1.7],[-.4,-1.7],
                     [-1.5,.1],[-.5,-.5],[-.15,-1.35]])
positive = -negative
assert np.all(negative.sum(axis=1) <= -1) and np.all(positive.sum(axis=1) >= 1)

fig, ax = plt.subplots(figsize=(8, 6))
plot_points(ax, negative, positive)
boundary(ax)
ax.scatter([-.5,.5],[-.5,.5], s=200, facecolors='none', edgecolors=INK,
           linewidths=1.3, label='Điểm hỗ trợ', zorder=5)
ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), fontsize=11)
ax.set_title('Hình học SVM: ranh giới và hai biên')
save(fig, 'svm_geometry')

# --- Candidate parallel boundaries (docs/01): cung huong khac offset; va khac huong. ---
cand_neg = np.array([[-1.6, -.6], [-.7, -1.3], [-1.1, -1.]])
cand_pos = np.array([[1.3, .7], [.6, 1.4], [1., 1.1]])


def line_for(direction, t, xs):
    return (t - direction[0]*xs)/direction[1]


direction = np.array([1., 1.])/np.sqrt(2)
proj_neg, proj_pos = cand_neg@direction, cand_pos@direction
offsets = {'A': -.3, 'B': (proj_neg.max()+proj_pos.min())/2, 'C': 1.3}
gaps = {name: min(t-proj_neg.max(), proj_pos.min()-t) for name, t in offsets.items()}
assert np.isclose(offsets['B'], 0., atol=1e-9)
assert gaps['B'] > gaps['A'] and gaps['B'] > gaps['C']

direction2 = np.array([1., .25]); direction2 = direction2/np.linalg.norm(direction2)
proj_neg2, proj_pos2 = cand_neg@direction2, cand_pos@direction2
t2 = (proj_neg2.max()+proj_pos2.min())/2
gap2 = min(t2-proj_neg2.max(), proj_pos2.min()-t2)
assert gaps['B'] > gap2

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
xline = np.linspace(-2.6, 2.6, 100)
ax = axes[0]
plot_points(ax, cand_neg, cand_pos)
for name, t in offsets.items():
    ax.plot(xline, line_for(direction, t, xline), color=INK if name == 'B' else GRAY,
            lw=2.4 if name == 'B' else 1.6, ls='-' if name == 'B' else '--',
            label=f'{name} (biên nhỏ nhất={gaps[name]:.2f})')
ax.set_title('Cùng hướng, khác offset')
ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=9.5)

ax2 = axes[1]
plot_points(ax2, cand_neg, cand_pos)
ax2.plot(xline, line_for(direction, offsets['B'], xline), color=INK, lw=2.4,
         label=f"Hướng tối ưu (margin={2*gaps['B']:.2f})")
ax2.plot(xline, line_for(direction2, t2, xline), color=RED, lw=1.8, ls='--',
         label=f'Hướng khác (margin={2*gap2:.2f})')
ax2.set_title('Khác hướng')
ax2.legend(loc='upper left', bbox_to_anchor=(1.03, 1), fontsize=9.5)
save(fig, 'candidate_boundaries')

# f(x) = x1 + x2; the red point has f(x)=0.6, y=+1, hence xi=0.4.
fig, ax = plt.subplots(figsize=(8,6))
plot_points(ax, negative, positive)
boundary(ax)
point = np.array([.1,.5]); projection = point + .2
assert np.isclose(point.sum(), .6) and np.isclose(projection.sum(), 1)
ax.scatter(*point, c=RED, marker='^', s=100, edgecolors=INK, linewidths=1.2, zorder=6)
ax.annotate('Điểm +1 trong biên\n$f(x)=0.6$, $\\xi=0.4$', xy=point,
            xytext=(.1,-1.1), fontsize=11, arrowprops={'arrowstyle':'->','color':INK},
            bbox={'boxstyle':'round,pad=.3','fc':'white','ec':'#cccccc'})
ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), fontsize=11)
ax.set_title('Soft margin: một điểm vi phạm biên')
save(fig, 'soft_margin')

# --- Hard vs soft margin (docs/02): mot outlier keo nghieng hard margin; soft margin voi vai C. ---
rng_hs = np.random.default_rng(7)
hs_neg = np.column_stack([rng_hs.normal(-2, .4, 10), rng_hs.normal(0, .6, 10)])
hs_pos = np.column_stack([rng_hs.normal(2, .4, 10), rng_hs.normal(0, .6, 10)])
hs_outlier = np.array([[.9, 0.]])  # nhãn -1, nằm sâu trong vùng của lớp +1
hs_X = np.vstack([hs_neg, hs_outlier, hs_pos])
hs_y = np.array([-1]*11 + [1]*10)

hard = SVC(kernel='linear', C=1e6).fit(hs_X, hs_y)
hard_margin = 2/np.linalg.norm(hard.coef_)
assert hard.score(hs_X, hs_y) == 1.0 and hard_margin < .15


def plot_boundary(ax, clf, color, lw, ls='-', label=None):
    xs = np.linspace(-3.4, 3.4, 200)
    w, b = clf.coef_[0], clf.intercept_[0]
    ax.plot(xs, -(w[0]*xs+b)/w[1], color=color, lw=lw, ls=ls, label=label, zorder=3)


fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
ax = axes[0]
ax.scatter(*hs_neg.T, c=BLUE, s=45, label='Lớp −1', zorder=4)
ax.scatter(*hs_outlier.T, c=BLUE, s=95, marker='s', edgecolors=INK, linewidths=1.4,
           label='Outlier nhãn −1', zorder=5)
ax.scatter(*hs_pos.T, c=RED, marker='^', s=50, label='Lớp +1', zorder=4)
plot_boundary(ax, hard, INK, 2.2, label=f'Hard margin (margin={hard_margin:.3f})')
ax.set(xlim=(-3.4,3.4), ylim=(-2.6,2.6), xlabel='$x_1$', ylabel='$x_2$',
       title='Hard margin bị một outlier kéo nghiêng')
ax.legend(loc='lower left', fontsize=9)

ax2 = axes[1]
ax2.scatter(*hs_neg.T, c=BLUE, s=45, zorder=4)
ax2.scatter(*hs_outlier.T, c=BLUE, s=95, marker='s', edgecolors=INK, linewidths=1.4, zorder=5)
ax2.scatter(*hs_pos.T, c=RED, marker='^', s=50, zorder=4)
soft_margins = {}
for C, color, ls in [(.1, GRAY, '--'), (1, BLUE, '-.'), (100, RED, '-')]:
    clf = SVC(kernel='linear', C=C).fit(hs_X, hs_y)
    soft_margins[C] = 2/np.linalg.norm(clf.coef_)
    plot_boundary(ax2, clf, color, 2.0, ls=ls, label=f'C={C} (margin={soft_margins[C]:.2f})')
assert soft_margins[.1] > soft_margins[1] > soft_margins[100]
ax2.set(xlim=(-3.4,3.4), ylim=(-2.6,2.6), xlabel='$x_1$', title='Soft margin: margin co lại khi C tăng')
ax2.legend(loc='lower left', fontsize=9)
save(fig, 'hard_vs_soft_margin')

# --- Logistic vs hinge loss (docs/02): tu "logistic" den "SVM" theo margin co dau. ---
z_lh = np.linspace(-2.2, 2.5, 400)
hinge_lh = np.maximum(0, 1-z_lh)
logistic_lh = np.log1p(np.exp(-z_lh))/np.log(2)  # co giãn để bằng 1 tại z=0 như hinge
assert np.isclose(np.log1p(np.exp(0))/np.log(2), 1.0)
assert np.isclose(np.log1p(np.exp(-1))/np.log(2), 0.4519, atol=1e-4)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(z_lh, hinge_lh, color=BLUE, lw=2.5, label='Hinge  $\\max(0,1-z)$')
ax.plot(z_lh, logistic_lh, color=RED, lw=2.5, ls='--', label='Logistic đã co giãn')
ax.axvline(1, color=GRAY, ls='--', lw=1)
ax.set(xlim=(-2.2,2.5), ylim=(-.08,3.3), xlabel='Margin có dấu  $z=y f(x)$',
       ylabel='Loss', title='Logistic loss và hinge loss theo margin có dấu')
ax.legend(loc='upper right', fontsize=11)
save(fig, 'logistic_vs_hinge')

fig, ax = plt.subplots(figsize=(7,5))
z = np.linspace(-1.2,2.5,400)
loss = np.maximum(0,1-z)
assert np.isclose(np.maximum(0, 1-.6), .4)
ax.plot(z, loss, color=BLUE, lw=2.5)
ax.scatter([.6,1],[.4,0], c=[RED,INK], s=50, zorder=4)
ax.annotate('$(0.6,\\,0.4)$', xy=(.6,.4), xytext=(.1,1.45),
            arrowprops={'arrowstyle':'->','color':GRAY}, fontsize=11)
ax.axvline(1,color=GRAY,ls='--',lw=1)
ax.set(xlim=(-1.2,2.5), ylim=(-.08,2.35), xlabel='Margin có dấu  $z=y f(x)$',
       ylabel='Hinge loss  $\\max(0,1-z)$',title='Hinge loss')
ax.set_xticks([-1,0,1,2]); ax.set_yticks([0,1,2])
save(fig, 'hinge_loss')

# Deterministic XOR clusters with the same samples and training C in both panels.
rng = np.random.default_rng(42)
centers = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
X = np.vstack([center + rng.normal(scale=.18, size=(20,2)) for center in centers])
y = np.repeat([1,-1,-1,1],20)
linear = SVC(kernel='linear', C=1).fit(X,y)
rbf = SVC(kernel='rbf', C=1, gamma=1).fit(X,y)
assert rbf.score(X,y) > .95
xx, yy = np.meshgrid(np.linspace(-1.8,1.8,250), np.linspace(-1.8,1.8,250))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
for ax, model, title in zip(axes, [linear,rbf], ['Linear SVM', 'RBF SVM']):
    score = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, score, levels=[score.min()-1,0,score.max()+1], colors=['#eaf0f8','#f9eeee'], alpha=.85)
    ax.contour(xx, yy, score, levels=[0], colors=INK, linewidths=1.8)
    ax.scatter(*X[y==-1].T, c=BLUE,s=22,label='Lớp −1',zorder=3)
    ax.scatter(*X[y==1].T, c=RED,marker='^',s=28,label='Lớp +1',zorder=3)
    ax.set(title=title,xlabel='$x_1$',xlim=(-1.8,1.8),ylim=(-1.8,1.8))
    ax.set_aspect('equal')
axes[0].set_ylabel('$x_2$')
fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(.5,-.04))
fig.suptitle('Cùng dữ liệu XOR tổng hợp; C = 1, RBF γ = 1',fontsize=13)
save(fig, 'linear_rbf_xor')

# --- RBF landmarks (docs/06): duong cong "giong nhau" va khong gian dac trung (phi1,phi2). ---
gamma_rbf, l1, l2 = 1.0, -1.5, 1.5
rbf_x = np.array([-2., -1., -.3, 0., .3, 1., 2.])
rbf_y = np.array([1, 1, -1, -1, -1, 1, 1])


def phi(x):
    return np.exp(-gamma_rbf*(x-l1)**2), np.exp(-gamma_rbf*(x-l2)**2)


phi1_pts, phi2_pts = phi(rbf_x)
sums = phi1_pts + phi2_pts
assert sums[rbf_y == -1].max() < 0.5 < sums[rbf_y == 1].min()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
ax = axes[0]
xs_fine = np.linspace(-3.2, 3.2, 400)
p1_fine, p2_fine = phi(xs_fine)
ax.plot(xs_fine, p1_fine, color=INK, lw=2.2, label='$\\phi_1=K(x,l_1)$')
ax.plot(xs_fine, p2_fine, color=GRAY, lw=2.2, ls='--', label='$\\phi_2=K(x,l_2)$')
ax.axvline(l1, color=INK, lw=1, ls=':'); ax.axvline(l2, color=GRAY, lw=1, ls=':')
ax.scatter(rbf_x[rbf_y == -1], np.full((rbf_y == -1).sum(), -.05), c=BLUE, s=55, zorder=5, label='Lớp −1')
ax.scatter(rbf_x[rbf_y == 1], np.full((rbf_y == 1).sum(), -.05), c=RED, marker='^', s=55, zorder=5, label='Lớp +1')
ax.set(xlim=(-3.2,3.2), ylim=(-.14,1.42), xlabel='$x$', ylabel='Giá trị kernel',
       title='Hai landmark $l_1=-1.5$, $l_2=1.5$')
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.0), fontsize=9, ncol=2)

ax2 = axes[1]
ax2.scatter(phi1_pts[rbf_y == -1], phi2_pts[rbf_y == -1], c=BLUE, s=80, label='Lớp −1', zorder=4)
ax2.scatter(phi1_pts[rbf_y == 1], phi2_pts[rbf_y == 1], c=RED, marker='^', s=80, label='Lớp +1', zorder=4)
diag = np.linspace(-.05, .85, 50)
ax2.plot(diag, .5-diag, color=INK, lw=1.6, ls='--', label='$\\phi_1+\\phi_2=0.5$')
ax2.set(xlim=(-.05,.85), ylim=(-.05,.85), xlabel='$\\phi_1$', ylabel='$\\phi_2$',
       title='Không gian đặc trưng $(\\phi_1,\\phi_2)$')
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=9.5)
save(fig, 'rbf_landmarks')

# --- Kernel comparison tren make_moons (docs/06): CV accuracy boxplot cho 4 kernel. ---
Xm, ym = make_moons(n_samples=200, noise=.25, random_state=0)
cv_moons = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
kernel_names = ['linear', 'poly', 'rbf', 'sigmoid']
cv_scores = {k: cross_val_score(make_pipeline(StandardScaler(), SVC(kernel=k, gamma='scale')), Xm, ym, cv=cv_moons)
             for k in kernel_names}
cv_means = {k: v.mean() for k, v in cv_scores.items()}
assert cv_means['rbf'] > cv_means['linear'] > cv_means['sigmoid']

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.boxplot([cv_scores[k] for k in kernel_names], tick_labels=kernel_names, patch_artist=True,
           medianprops={'color': INK, 'lw': 1.8}, boxprops={'facecolor': '#eaf0f8', 'edgecolor': INK})
jitter_rng = np.random.default_rng(1)
best_kernel = max(cv_means, key=cv_means.get)
for i, k in enumerate(kernel_names):
    jitter = jitter_rng.uniform(-.08, .08, len(cv_scores[k]))
    ax.scatter(np.full(len(cv_scores[k]), i+1)+jitter, cv_scores[k],
               c=RED if k == best_kernel else GRAY, s=14, alpha=.6, zorder=3)
ax.set(ylabel='Accuracy (CV)', xlabel='Kernel',
       title='So sánh kernel trên make_moons (50 lần CV mỗi kernel)')
save(fig, 'kernel_comparison_moons')

# --- KKT alpha number line (docs/04): ba vùng của alpha_i trên [0,C]. ---
# alpha_1=alpha_2=0.25 va alpha_3=alpha_4=0 la nghiem cua vi du giai tay (docs/07),
# duoc nhac lai trong docs/04 muc 4.
C_kkt = 1.0
alpha_free, alpha_zero = 0.25, 0.0
assert alpha_zero == 0.0
assert 0 < alpha_free < C_kkt

fig, ax = plt.subplots(figsize=(9.5, 4))
ax.axhline(0, color=GRAY, lw=1.6, zorder=1, xmin=0.02, xmax=0.98)
ax.plot([0, C_kkt], [0, 0], color=BLUE, lw=5, zorder=2, solid_capstyle='butt')
ax.scatter([0], [0], s=170, color=GRAY, zorder=5, edgecolors=INK, linewidths=1.2)
ax.scatter([C_kkt], [0], s=170, color=RED, marker='s', zorder=5, edgecolors=INK, linewidths=1.2)
ax.scatter([alpha_free], [0], s=90, color=INK, marker='^', zorder=6)

ax.annotate('$\\alpha_i=0$\n$y_if(x_i)\\geq1$\nkhông phải SV',
            xy=(0, 0), xytext=(-0.02, 1.55), fontsize=10.5, ha='center',
            arrowprops={'arrowstyle': '->', 'color': INK},
            bbox={'boxstyle': 'round,pad=.35', 'fc': 'white', 'ec': GRAY})
ax.annotate('$0<\\alpha_i<C$\n$y_if(x_i)=1$\nSV tự do trên biên',
            xy=(C_kkt / 2, 0), xytext=(C_kkt / 2, 1.55), fontsize=10.5, ha='center',
            arrowprops={'arrowstyle': '->', 'color': INK},
            bbox={'boxstyle': 'round,pad=.35', 'fc': 'white', 'ec': BLUE})
ax.annotate('$\\alpha_i=C$\n$y_if(x_i)\\leq1,\\ \\xi_i\\geq0$\ntrong biên / sai',
            xy=(C_kkt, 0), xytext=(C_kkt + 0.02, 1.55), fontsize=10.5, ha='center',
            arrowprops={'arrowstyle': '->', 'color': INK},
            bbox={'boxstyle': 'round,pad=.35', 'fc': 'white', 'ec': RED})

ax.annotate('$\\alpha_1=\\alpha_2=0.25$', xy=(alpha_free, 0), xytext=(alpha_free, -1.0),
            fontsize=10, ha='center', color=INK,
            arrowprops={'arrowstyle': '->', 'color': INK})
ax.annotate('$\\alpha_3=\\alpha_4=0$', xy=(0, 0), xytext=(0.17, -1.5),
            fontsize=10, ha='left', color=GRAY,
            arrowprops={'arrowstyle': '->', 'color': GRAY})
ax.text(0, -0.45, '0', fontsize=12, ha='center', color=INK)
ax.text(C_kkt, -0.45, '$C$', fontsize=12, ha='center', color=INK)

ax.set_xlim(-0.35, C_kkt + 0.35)
ax.set_ylim(-1.9, 2.9)
ax.axis('off')
ax.set_title('Ba vùng của $\\alpha_i$ trên $[0,C]$', pad=18)
save(fig, 'kkt_alpha_number_line')

# --- SMO feasible box (docs/05): hinh vuong [0,C]x[0,C], duong rang buoc, L/H, clip. ---
C_smo = 1.0


def lh_bounds(ai, aj, same_sign, C):
    if same_sign:
        return max(0.0, ai + aj - C), min(C, ai + aj)
    return max(0.0, aj - ai), min(C, C + aj - ai)


def draw_smo_panel(ax, ai_old, aj_old, aj_new_unclipped, same_sign, title):
    C = C_smo
    L, H = lh_bounds(ai_old, aj_old, same_sign, C)
    aj_clipped = min(H, max(L, aj_new_unclipped))
    sign = 1.0 if same_sign else -1.0
    const = ai_old + sign * aj_old  # alpha_i + sign*alpha_j = const (duong rang buoc)

    def ai_of(aj):
        return const - sign * aj

    # doan kha thi cua duong rang buoc trong hop, ung voi alpha_j trong [L,H]
    aj_seg = np.linspace(L, H, 50)
    ai_seg = ai_of(aj_seg)
    assert np.all(ai_seg >= -1e-9) and np.all(ai_seg <= C + 1e-9)

    ax.add_patch(plt.Rectangle((0, 0), C, C, fill=False, edgecolor=INK, lw=1.8, zorder=2))
    aj_ext = np.linspace(-0.45 * C, 1.45 * C, 60)
    ax.plot(ai_of(aj_ext), aj_ext, color=GRAY, lw=1.3, ls=':', zorder=1)
    ax.plot(ai_seg, aj_seg, color=BLUE, lw=3.5, zorder=3, label='Đoạn khả thi')
    ax.axhline(L, color=GRAY, lw=1.1, ls='--', zorder=1)
    ax.axhline(H, color=GRAY, lw=1.1, ls='--', zorder=1)
    for value, name in [(L, '$L$'), (H, '$H$')]:
        ax.text(1.47 * C, value, name, fontsize=11, va='center', color=INK,
                bbox={'fc': 'white', 'ec': 'none', 'pad': 1}, zorder=4)

    ax.scatter([ai_old], [aj_old], s=75, color=INK, zorder=5, label='$(\\alpha_i,\\alpha_j)$ cũ')
    ax.scatter([ai_of(aj_new_unclipped)], [aj_new_unclipped], s=75, color=RED, marker='x',
               linewidths=2.4, zorder=5, label='$\\alpha_j^{new}$ chưa clip')
    ax.scatter([ai_of(aj_clipped)], [aj_clipped], s=90, facecolors='none', edgecolors=RED,
               linewidths=1.8, zorder=5, label='$\\alpha_j^{new}$ đã clip')
    ax.annotate('', xy=(ai_of(aj_clipped), aj_clipped), xytext=(ai_of(aj_new_unclipped), aj_new_unclipped),
                arrowprops={'arrowstyle': '->', 'color': RED, 'lw': 1.3})

    ax.set(xlim=(-0.45 * C, 1.6 * C), ylim=(-0.45 * C, 1.45 * C),
           xlabel='$\\alpha_i$', ylabel='$\\alpha_j$', title=title)
    ax.set_aspect('equal')
    return L, H, aj_clipped


# y_i != y_j: L=max(0,aj-ai), H=min(C,C+aj-ai)
ai1, aj1, aj1_new = 0.8, 0.3, 0.7
L1, H1 = lh_bounds(ai1, aj1, same_sign=False, C=C_smo)
assert (L1, H1) == (0.0, 0.5)
assert min(H1, max(L1, aj1_new)) == 0.5

# y_i == y_j: L=max(0,ai+aj-C), H=min(C,ai+aj)
ai2, aj2, aj2_new = 0.3, 0.3, -0.2
L2, H2 = lh_bounds(ai2, aj2, same_sign=True, C=C_smo)
assert (L2, H2) == (0.0, 0.6)
assert min(H2, max(L2, aj2_new)) == 0.0

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
draw_smo_panel(axes[0], ai1, aj1, aj1_new, same_sign=False, title='$y_i\\neq y_j$')
draw_smo_panel(axes[1], ai2, aj2, aj2_new, same_sign=True, title='$y_i=y_j$')
axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
save(fig, 'smo_lh_box')

# --- 4-point hand example (docs/07): w, b, boundary, margins, SV, margin width. ---
hand_X = np.array([[-1., -1.], [1., 1.], [-2., -1.], [2., 1.]])
hand_y = np.array([-1, 1, -1, 1])
hand_w, hand_b = np.array([0.5, 0.5]), 0.0
hand_f = hand_X @ hand_w + hand_b
assert np.allclose(hand_f, [-1.0, 1.0, -1.5, 1.5])
assert np.allclose(hand_y * hand_f, [1.0, 1.0, 1.5, 1.5])
margin_width = 2 / np.linalg.norm(hand_w)
assert np.isclose(margin_width, 2 * np.sqrt(2))
assert np.isclose(np.linalg.norm(hand_X[1] - hand_X[0]), margin_width)  # x2-x1 // w, do đúng trên 2 biên

hand_clf = SVC(kernel='linear', C=1).fit(hand_X, hand_y)
assert np.allclose(hand_clf.coef_[0], hand_w, atol=1e-6)
assert np.isclose(hand_clf.intercept_[0], hand_b, atol=1e-6)
assert sorted(hand_clf.support_.tolist()) == [0, 1]

fig, ax = plt.subplots(figsize=(7.5, 6.5))
plot_points(ax, hand_X[hand_y == -1], hand_X[hand_y == 1])
x1_range = np.linspace(-2.8, 2.8, 200)
ax.plot(x1_range, -x1_range, color=INK, lw=2, label='$f(x)=0$', zorder=2)
ax.plot(x1_range, 2 - x1_range, color=GRAY, lw=1.5, ls='--', label='$f(x)=\\pm1$', zorder=2)
ax.plot(x1_range, -2 - x1_range, color=GRAY, lw=1.5, ls='--', zorder=2)
ax.scatter(*hand_X[:2].T, s=230, facecolors='none', edgecolors=INK, linewidths=1.4,
           label='Support vector', zorder=5)
for label, xy, off in [('$x_1$', hand_X[0], (-.32, -.12)), ('$x_2$', hand_X[1], (.14, .14)),
                       ('$x_3$', hand_X[2], (-.34, -.12)), ('$x_4$', hand_X[3], (.14, .14))]:
    ax.annotate(label, xy=xy, xytext=xy + off, fontsize=12, color=INK)
ax.annotate('', xy=tuple(hand_w), xytext=(0, 0),
            arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': 2.6}, zorder=6)
ax.annotate('$w=(0.5,0.5)$', xy=tuple(hand_w), xytext=(hand_w[0] + .12, hand_w[1] - .25),
            fontsize=11, color=BLUE)
shift = np.array([1., -1.]) / np.sqrt(2) * 1.3
p1, p2 = hand_X[0] + shift, hand_X[1] + shift
ax.annotate('', xy=tuple(p2), xytext=tuple(p1),
            arrowprops={'arrowstyle': '<->', 'color': RED, 'lw': 1.6}, zorder=6)
mid = (p1 + p2) / 2
ax.text(mid[0] + .18, mid[1], '$2\\sqrt{2}\\approx2.83$', color=RED, fontsize=11)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11)
ax.set_title('Ví dụ giải tay: 4 điểm, w, b và hai biên')
save(fig, 'hand_example')

print(f'Saved 12 figures in {OUT}; XOR training accuracy: linear={linear.score(X,y):.3f}, RBF={rbf.score(X,y):.3f}')
