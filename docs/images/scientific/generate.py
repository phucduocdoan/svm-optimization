"""Tạo hình minh họa SVM. Chạy: python docs/images/scientific/generate.py."""
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
        fig.savefig(OUT / f'{name}.{suffix}', dpi=300, bbox_inches='tight')
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
print(f'Saved 4 figures in {OUT}; XOR training accuracy: linear={linear.score(X,y):.3f}, RBF={rbf.score(X,y):.3f}')
