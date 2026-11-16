"""Tạo hình minh họa SVM. Chạy: uv run python docs/images/scientific/generate.py."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

print(f'Saved 7 figures in {OUT}; XOR training accuracy: linear={linear.score(X,y):.3f}, RBF={rbf.score(X,y):.3f}')
