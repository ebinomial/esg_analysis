# ESG ба санхүүгийн шинжилгээ (Company Panel)

Нэг датасет дээр гурван даалгавар гүйцэтгэсэн: 1,000 компанийг 2015-2025 он (11 жил)
хүртэл жил тутам ажигласан balanced panel. Санхүүгийн үзүүлэлт (revenue, profit margin,
market cap, growth), дөрвөн ESG оноо (overall + environmental/social/governance pillar),
болон байгаль орчны footprint (carbon, water, energy) багтана. Датасетыг notebook бүр
дотроос Hugging Face Hub-аас шууд татаж авна:

```python
from datasets import load_dataset
df = load_dataset("margiela00/task_dataset")["train"].to_pandas()
```

Бүх deliverable нь `scripts/` дотор Jupyter notebook хэлбэрээр байна. `scripts/`-ээс гадна
байгаа бүх зүйл submission-д ороогүй, зөвхөн энэ `README.md` болон `.gitignore` хамаарна.

## Гурван даалгаварт ижил method-р ажилласан

Notebook бүр адилхан протоколоор явна. Ингэснээр үр дүнгүүд харьцуулах боломжтой,
шударга болно:

1. Богино, өргөн panel-д тохирсон feature engineering (11 жил гэдэг seasonal univariate
   аргад хэт богино тул cross-sectional / feature-based хандлагыг илүүд үзсэн).
2. Crude baseline загвар ба илүү комплекс загвар.
3. Хоёр загварын хооронд hypothesis test хийж, нэмэлт complexity үнэхээр үр өгөөжтэй
   бол л түүнийг сонгоно (Occam's razor зарчмын дагуу).

Дүгнэлтүүд нь notebook бүрийн төгсгөлийн markdown cell-д байрлаж байгаа. Нэмээд би өөрийнхөө тооцооллын үед бодож байсан дүгнэлтүүдийг дунд нь cell-д бичсэн байгаа.

## Тохиргоо (Setup)

```bash
# shine environment
python -m venv .venv
.venv/Scripts/activate 
# package version neeh hamaagui
pip install datasets pandas numpy scikit-learn lightgbm scipy seaborn matplotlib jupyter
```

Дараа нь `scripts/` доторх notebook-уудыг нээж, тус бүрийг дээрээс доош дараалан
ажиллуулна. Notebook бүр бие даасан: датасетыг өөрөө дахин ачаалдаг тул хоорондоо
төлөв (state) хуваалцдаггүй.

## `scripts/` доторх notebook-ууд (энэ дарааллаар ажиллуулах)

Дараалал нь унших дараалал болохоос data dependency биш (тус бүр standalone ажиллана).
Unsupervised бүтэц -> forecasting -> supervised risk labeling гэсэн урсгалаар явна.

### 1. `clustering.ipynb` - компанийн segmentation (unsupervised)
Компани бүрийн 11 жилийн түүхийг 46 feature болгон хураана (level, trend, dispersion, мөн
хатуу эерэг цувааны хувьд CAGR/volatility), log + standardize хийж, PCA-аар 15 хэмжээст
(95% variance) бууруулаад KMeans-ээр cluster хийнэ. K-г silhouette ба inertia-гаар сонгоно.
Үр дүн: K=5 тайлбарлахуйц archetype, эдгээр нь carbon intensity-д тулгуурласан илүү цэвэр
2 талт (ногоон/бор) хуваалтын дотор үүрлэнэ. Segment-ууд нь зүгээр салбарын хуулбар биш
(ARI vs industry ~ 0.13). Үүнийг эхэлж ажиллуул: feature engineering ба EDA нь нөгөө хоёрын
context-ыг тавьж өгнө.

### 2. `forecasting.ipynb` - market cap-ийн forecasting (supervised regression)
`market_cap`-ийг (хүчтэй right skew тул log хувиргасан) бүх компани дээр сургасан нэг global
LightGBM-ээр (cross-learning) таамаглаж, 2025 он дээр нэг алхам урагш (1-step ahead) тестэлнэ.
Baseline M0 (market_cap-ийн 3 lag) ба комплекс M1 (бүх feature-ийн lag + статик талбар)-ыг
харьцуулна; hyperparameter ижил тул тест нь зөвхөн feature-ийн нөлөөг тусгаарлана. Компани
тус бүрийн алдаа дээрх Wilcoxon signed-rank тест M1-ийг статистикийн хувьд илүү (p ~ 5e-6) гэж
гаргасан ч median APE дөнгөж ~1 нэгж хувиар л сайжирсан. Шийдвэр: энгийн M0-г сонгоно.

### 3. `classification.ipynb` - ESG эрсдэлийн classification (supervised)
Risk label-ыг хамгийн сүүлийн жилийн (2025) ESG overall оноогоор үүсгэнэ: оноо < 50 бол
`risky` (LSEG/Refinitiv-ийн below-average grade зааг), ингэснээр 298 risky vs 702 not-risky
болно. Гурван ESG pillar оноог target leakage тул хасаж, зөвхөн financial + footprint feature
ашиглана. Soft-margin RBF SVM (class-balanced)-ийг recall-first байдлаар үнэлнэ. Ablation
харуулахад financial дан ганцаараа санамсаргүйтэй ойролцоо (ROC-AUC 0.55), харин footprint
нэмэхэд 0.70 болж, risky class дээр 0.72 recall өгнө.

## Ерөнхий дүгнэлт

- Segmentation нь байгаль орчны ("ногоон vs бор") тэнхлэг дээр таван тайлбарлахуйц archetype
  гаргасан ч cluster-ууд сул (silhouette ~ 0.14) тул шударга тайлбар нь "өргөн 2 талт
  хуваалтын дотор таван segment", цэвэр тусгаарлагдсан cluster биш.
- Forecasting: комплекс загвар статистикийн хувьд илүү боловч практик талдаа тэнцүү. Аль
  хэдийн ~31% алдаатай forecast дээр ~1 нэгж хувийн median ахиц ямар ч шийдвэрийг өөрчлөхгүй
  тул энгийн загвар хожино.
- ESG эрсдэл нь ашигтай screen хэлбэрээр илэрхийлэгдэнэ (ROC-AUC 0.70, risky recall 0.72),
  гэвч signal нь бараг бүхэлдээ environmental footprint-оос ирдэг ба энэ нь өөрөө environment
  score-ийн proxy. Санхүүгийн суурь үзүүлэлт бие даасан эрсдэлийн signal агуулаагүй.
- Гол санаа: гурван комплекс загварын хоёр нь өөрсдийн complexity-г зөвтгөж чадсангүй. Үүнийг
  hypothesis test-ээр баталж, шулуухан мэдээлэх нь энэ ажлын гол зорилго.
