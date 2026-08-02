"""Builds a self-contained presentation.html (arrow keys / click to advance)."""
import base64, json, pathlib

R = json.load(open("results.json"))

def img(name):
    b = base64.b64encode(open(f"figures/{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"

def slide(speaker, kicker, head, fig=None, points=None, big=None, note=None):
    s = ['<section class="slide%s">' % ("" if fig else " text")]
    if speaker:
        s.append(f'<div class="chip">{speaker}</div>')
    if kicker:
        s.append(f'<p class="kicker">{kicker}</p>')
    s.append(f"<h2>{head}</h2>")
    if points:
        s.append("<ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>")
    if big:
        s.append(f'<p class="big">{big}</p>')
    if fig:
        s.append(f'<div class="fig"><img src="{img(fig)}" alt=""></div>')
    if note:
        s.append(f'<p class="note">{note}</p>')
    s.append("</section>")
    return "\n".join(s)

SLIDES = []

SLIDES.append(f"""<section class="slide title">
<p class="kicker">STAT 451 &middot; Group Project</p>
<h1>Which Measurements<br>Diagnose a Breast Mass?</h1>
<p class="sub">Working out which of 30 nuclear measurements actually matter</p>
<p class="names">Amoolya &middot; Bonnie &middot; Dhruvika &middot; Kalynn &middot; Edward</p>
<p class="note">Breast Cancer Wisconsin (Diagnostic) &middot; {R['n']} masses &middot; UCI ML Repository</p>
</section>""")

SLIDES.append(slide(
    "Amoolya", "The question", "Our question and our data",
    points=[
        "In a fine-needle breast biopsy, the sample is photographed and "
        "software measures the <b>cell nuclei</b>.",
        f"<b>{R['n']} masses</b>, {R['pct_malignant']}% malignant. Ten "
        "measurements, each reported three ways (mean, standard error, and a "
        "&ldquo;worst&rdquo; value), so 30 features.",
        "<b>Which measurements actually separate malignant from benign, and "
        "how few do you need?</b>",
    ],
    note=f"We split the data once, 80/20 and stratified. Everything after that "
         f"(model, hyperparameters, feature ranking, cutoff) came from 5-fold "
         f"cross-validation on the {R['n_train']} training masses. We left the "
         f"{R['n_test']} test masses alone until the end."))

SLIDES.append(slide(
    "Amoolya", "What the data shows",
    "Size and shape do most of the work",
    fig="fig2_curves.png"))

SLIDES.append(slide(
    "Edward", "A problem", "The features repeat each other",
    fig="fig1_redundancy.png",
    note=f"{R['pairs_r90']} of the {R['n_pairs']} pairs correlate above 0.90. "
         f"Radius, perimeter and area are the same measurement three times over "
         f"(r = {R['max_r']})."))

SLIDES.append(slide(
    "Edward", "Why it matters", "So we did not trust the coefficients",
    fig="fig6_stability.png",
    note="We refit LASSO on 200 bootstrap resamples. Some size feature is kept "
         "every time, but which one is close to random. If we had just read one "
         "fitted model, we would have picked whichever it happened to keep."))

SLIDES.append(slide(
    "Bonnie", "Model comparison", "Which model should we use?",
    fig="fig4_models.png",
    note=f"All tuned by cross-validation on the training set. The RBF-SVM wins "
         f"by {R['best_cv_auc'] - 0.9959:.4f} AUC over a linear SVM, which is "
         f"not a difference a patient would notice. The decision tree is the "
         f"one that lags: its splits are axis-aligned while the two groups "
         f"separate along a diagonal, and a single tree varies a lot from fold "
         f"to fold."))

SLIDES.append(slide(
    "Dhruvika", "Cutting it down", "How many features do we actually need?",
    fig="fig5_howfew.png",
    note=f"Principal components (uncorrelated axes sorted by how much variance "
         f"they explain) do slightly better, but by at most 0.009 AUC. You also "
         f"need {R['pc_for_95']} of them for 95% of the variance, and none is "
         f"something a pathologist could measure."))

SLIDES.append(slide(
    "Dhruvika", "A surprise", "One result we did not expect",
    points=[
        "<b>worst texture</b> is kept in <b>100%</b> of resamples and comes "
        "<b>first</b> on permutation importance.",
        "But on its own it ranks only <b>17th of 30</b> (AUC 0.79).",
        "We think that is because it is almost uncorrelated with size "
        "(r&nbsp;=&nbsp;0.37).",
    ],
    big="What matters is whether a feature adds something new, not how strong "
        "it looks on its own."))

SLIDES.append(slide(
    "Kalynn", "Two features at once", "Putting two measurements together",
    fig="fig3_surface.png",
    note=f"Worst radius and worst smoothness, r = {R['pair_r']}, "
         f"cross-validated AUC {R['pair_auc']:.3f}. A big nucleus is malignant "
         f"whatever its smoothness. Smoothness only changes the answer for "
         f"mid-sized ones."))

SLIDES.append(slide(
    "Kalynn", "Choosing the cutoff",
    "Missing a cancer is worse than a false alarm",
    fig="fig7_test.png",
    note=f"So we set the cutoff for {R['target_sens']}% sensitivity in training "
         f"cross-validation instead of going for the best accuracy. On the test "
         f"set we caught <b>{R['tp']} of {R['tp']+R['fn']}</b> malignant "
         f"masses, at {R['test_spec']}% specificity and {R['test_acc']}% "
         f"accuracy, against {R['baseline_acc']}% for guessing benign every "
         f"time. Training AUC {R['train_auc_best']:.3f} vs test "
         f"{R['test_auc']:.3f}, so it does not look overfit."))

SLIDES.append(slide(
    "All five", "Conclusion", "What we found",
    points=[
        "Malignant nuclei are <b>bigger</b>, <b>more deeply indented</b> and "
        "<b>more unevenly textured</b>.",
        f"A measurement&rsquo;s <b>worst</b> reading beats its <b>average</b> "
        f"({R['group_power']['worst']:.2f} vs "
        f"{R['group_power']['mean']:.2f} average AUC).",
        f"<b>Six</b> measurements do about as well as thirty "
        f"({R['auc_at_K']:.3f} vs {R['full30_auc']:.3f} AUC).",
        "<b>Weaknesses:</b> one lab, early 1990s, hand-picked and "
        "human-segmented samples. Only 114 test cases, so specificity is good "
        "to about &plusmn;5 points.",
    ],
    big="Where you set the cutoff is really a medical decision, not a "
        "statistical one."))

DECK = "\n".join(SLIDES)
N = len(SLIDES)

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which Measurements Diagnose a Breast Mass? &mdash; STAT 451</title>
<style>
  :root {{ --ink:#111; --ink2:#55534e; --rule:#dcdad4; --surface:#fdfdfc;
           --accent:#1c5cab; --warm:#e34948; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html,body {{ height:100%; background:#e9e7e1; }}
  body {{ font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
    Helvetica,Arial,sans-serif; color:var(--ink); }}
  .deck {{ height:100%; }}
  .slide {{ display:none; position:relative; width:100%; height:100%;
    background:var(--surface); padding:5.5vh 6vw 8vh;
    flex-direction:column; }}
  .slide.on {{ display:flex; }}
  .kicker {{ font-size:1.55vh; letter-spacing:2.2px; text-transform:uppercase;
    color:var(--accent); font-weight:700; margin-bottom:1.1vh; }}
  h1 {{ font-size:7.4vh; line-height:1.08; letter-spacing:-1.6px;
    font-weight:700; }}
  h2 {{ font-size:4.3vh; line-height:1.16; letter-spacing:-.7px;
    font-weight:650; max-width:22ch; }}
  .slide.title, .slide.text {{ justify-content:center; }}
  .slide.text ul {{ max-width:76ch; }}
  .slide.text li {{ font-size:2.85vh; margin-bottom:2.9vh; }}
  .slide.text .note {{ margin-top:3.4vh; padding-top:2.4vh;
    border-top:1px solid var(--rule); }}
  .title .sub {{ font-size:3vh; color:var(--ink2); margin-top:2.4vh;
    font-style:italic; }}
  .title .names {{ font-size:2.5vh; margin-top:5.5vh; font-weight:600; }}
  .title .note {{ margin-top:1.4vh; }}
  ul {{ margin:3.4vh 0 0 0; list-style:none; max-width:56ch; }}
  li {{ font-size:2.55vh; line-height:1.5; margin-bottom:2.4vh;
    padding-left:2.6vh; position:relative; }}
  li::before {{ content:""; position:absolute; left:0; top:1.05em;
    width:1.1vh; height:1.1vh; border-radius:50%; background:var(--accent); }}
  .big {{ font-size:3.1vh; line-height:1.35; margin-top:3.6vh;
    padding-left:2.2vh; border-left:4px solid var(--warm); font-weight:600;
    max-width:34ch; }}
  .fig {{ flex:1; min-height:0; margin-top:2.4vh; display:flex;
    align-items:center; justify-content:center; }}
  .fig img {{ max-width:100%; max-height:100%; object-fit:contain; }}
  .note {{ font-size:1.95vh; line-height:1.5; color:var(--ink2);
    margin-top:2.2vh; max-width:88ch; }}
  .note b {{ color:var(--ink); }}
  .chip {{ position:absolute; top:4.4vh; right:6vw; font-size:1.7vh;
    font-weight:700; letter-spacing:.6px; color:#fff; background:var(--ink);
    padding:.55vh 1.5vh; border-radius:99px; }}
  #bar {{ position:fixed; left:0; bottom:0; height:5px; background:var(--accent);
    transition:width .2s ease; z-index:5; }}
  #num {{ position:fixed; right:1.6vw; bottom:1.6vh; font-size:1.7vh;
    color:var(--ink2); z-index:5; font-variant-numeric:tabular-nums; }}
  @media print {{
    .slide {{ display:flex !important; page-break-after:always;
      height:100vh; }} #bar,#num {{ display:none; }}
  }}
</style></head><body>
<div class="deck" id="deck">{DECK}</div>
<div id="bar"></div><div id="num"></div>
<script>
  const s = [...document.querySelectorAll('.slide')];
  let i = 0;
  function show(n) {{
    i = Math.max(0, Math.min(s.length - 1, n));
    s.forEach((el, k) => el.classList.toggle('on', k === i));
    document.getElementById('bar').style.width =
      ((i + 1) / s.length * 100) + '%';
    document.getElementById('num').textContent = (i + 1) + ' / ' + s.length;
    location.hash = i + 1;
  }}
  addEventListener('keydown', e => {{
    if (['ArrowRight',' ','PageDown','n'].includes(e.key)) {{ show(i+1); e.preventDefault(); }}
    if (['ArrowLeft','PageUp','p'].includes(e.key)) {{ show(i-1); e.preventDefault(); }}
    if (e.key === 'Home') show(0);
    if (e.key === 'End') show(s.length - 1);
  }});
  addEventListener('click', e => show(i + (e.clientX < innerWidth * 0.25 ? -1 : 1)));
  show(parseInt(location.hash.slice(1)) - 1 || 0);
</script></body></html>"""

pathlib.Path("presentation.html").write_text(HTML, encoding="utf-8")
print(f"presentation.html written: {N} slides")
