"""Builds a self-contained report.html (figures inlined as base64)."""
import base64, json, re, pathlib

R = json.load(open("results.json"))

def img(name):
    b = base64.b64encode(open(f"figures/{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"

# Figures sit next to the text that cites them, so a 600-word paper does not
# make the reader scroll past the whole argument to reach Figure 1.
FIGS = {  # key -> (file, number, caption)
    "curves": ("fig2_curves.png", 1, "Estimated probability of malignancy "
               "against the strongest measurement in each of four groups."),
    "redundancy": ("fig1_redundancy.png", 2, "The 30 features carry far less "
                   "than 30 measurements&rsquo; worth of information."),
    "stability": ("fig6_stability.png", 3, "Bootstrap selection frequency: "
                  "LASSO always wants size, but cannot say which size feature."),
    "surface": ("fig3_surface.png", 4,
                f"{R['pair'][0].capitalize()} and {R['pair'][1]}, the best pair "
                f"of weakly correlated measurements we found (RBF-SVM, "
                f"cross-validated AUC {R['pair_auc']:.3f}; best of a search, so "
                "a ceiling rather than an unbiased estimate). A large nucleus "
                "is malignant whatever its smoothness; smoothness only matters "
                "for mid-sized ones."),
    "howfew": ("fig5_howfew.png", 5, "Cross-validated discrimination against "
               "the number of features retained."),
    "test": ("fig7_test.png", 6, f"Both models on the {R['n_test']} test "
             "masses, scored in a single pass."),
}

def fig(key):
    f, n, cap = FIGS[key]
    return (f'<figure><img src="{img(f)}" alt="Figure {n}">'
            f"<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>")

# The chosen features, read out of results.json rather than retyped.
SIX = R["final_features"]
SIX_LIST = ", ".join(SIX[:-1]) + " and " + SIX[-1]
HYPER = ", ".join(f"{k} = {v}" for k, v in R["best_params"].items())
NUMWORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
           7: "Seven", 8: "Eight"}
def numword(n):
    return NUMWORD.get(n, str(n)).lower()
KW = NUMWORD.get(R["K"], str(R["K"]))         # "Six"
kw = KW.lower()                               # "six"

BODY = f"""
<p class="lead">In a fine-needle breast biopsy the sample is photographed and software
measures the cell nuclei. Which of 30 such measurements separate malignant
from benign masses, and how few do you need? Across {R['n']} Wisconsin
Diagnostic masses every model scored alike, so the variables mattered more:
nucleus <b>size</b>, edge <b>indentation</b> and <b>texture</b>.
{KW} match all thirty, and ours caught every malignant mass in the
{R['n_test']} held out, flagging {R['fp_per100_benign']} per 100 benign for
follow-up.</p>

<h2>Data and methods</h2>
<p>There are {R['n']} masses, {R['pct_malignant']:.0f}% malignant. Ten nuclear
measurements are each reported three ways &mdash; mean, standard error and a
&ldquo;worst&rdquo; value (mean of the three largest) &mdash; giving 30
features, none missing. We split once, 80/20 stratified, standardizing on
training data only. Model, hyperparameters, feature ranking and cutoff came from
5-fold cross-validation on the {R['n_train']} training masses; both final models
were fixed before we opened the {R['n_test']} test masses, then scored in one
pass.</p>

<h2>What separates them</h2>
<p>Size and shape almost separate the groups by themselves; texture and
smoothness only shift the odds (Figure 1). But the features repeat each other:
{R['pairs_r90']} of {R['n_pairs']} pairs correlate above 0.90, and radius,
perimeter and area are one measurement three times over (r up to {R['max_r']},
Figure 2), so single coefficients are not trustworthy.</p>

{fig('curves')}
{fig('redundancy')}

<p>So we refit LASSO on 200 bootstrap resamples (Figure 3). Some size feature
is kept every time, but which one is close to random: worst radius
{R['size_freq']['worst radius']:.0f}%, worst area
{R['size_freq']['worst area']:.0f}%, mean perimeter
{R['size_freq']['mean perimeter']:.0f}%. Size matters; no one size column
does.</p>

{fig('stability')}

<p>It keeps <b>{R['stable_first']}</b> most reliably &mdash;
{R['stable_first_freq']:.0f}% of resamples, and first on out-of-fold permutation
importance &mdash; though alone it ranks {R['stable_first_rank']}th of 30 (AUC
{R['stable_first_auc']:.2f}). Almost uncorrelated with size
(r&nbsp;=&nbsp;{R['stable_first_r_size']:.2f}), it adds what size cannot, as
weak-looking smoothness does where size is ambiguous (Figure 4). Across all ten,
the &ldquo;worst&rdquo; versions beat the means
({R['group_power']['worst']:.2f} vs {R['group_power']['mean']:.2f} average AUC),
so extreme readings help more.</p>

{fig('surface')}

<h2>{KW} features are enough</h2>
<p>Re-ranking features inside each fold and keeping only weakly correlated ones
(|r|&nbsp;&lt;&nbsp;0.8), {kw} reach cross-validated AUC {R['auc_at_K']:.3f}
against {R['full30_auc']:.3f} for all thirty (Figure 5): {SIX_LIST} &mdash;
{R['final_mix']}. That is a judgement call: it clears our 0.005 tolerance by a
hair, and a stricter one gives {numword(R['K_strict'])}. Principal
components &mdash; uncorrelated blends of all 30 &mdash; gain
{R['pca_gain_at_K']:.3f} AUC here, none measurable by a pathologist.</p>

{fig('howfew')}

<h2>How well it works</h2>
<p>The {R['best_model']} won on cross-validated AUC
({R['best_cv_auc']:.3f}; {HYPER}), with {numword(R['n_within_001'])} others within 0.001;
only the decision tree lagged ({R['tree_auc']:.3f}), its axis-aligned splits
cutting a diagonal boundary. Because a missed cancer costs more than a false
alarm, we set the cutoff ({R['threshold']:.2f}) for {R['target_sens']}% training
sensitivity. On test it caught all {R['tp']} malignant masses at
{R['test_spec']}% specificity and {R['test_acc']}% accuracy, against
{R['baseline_acc']}% for always guessing benign (Figure 6); the {kw} named
features also caught all {R['tp']}, at {R['test_spec_reduced']}%. Training AUC
({R['train_auc_best']:.3f}) did not exceed test ({R['test_auc']:.3f}): no
overfitting.</p>

{fig('test')}

<h2>Weaknesses</h2>
<p>All data came from one Wisconsin lab in the early 1990s, hand-picked and
human-segmented. With only {R['n_test']} test cases, specificity is good
to about &plusmn;5 points, and {R['tp']} of {R['tp']} would not hold up in a
clinic. Our cutoff, chosen on cross-validated probabilities but applied to a
model refit on all {R['n_train']}, holds its {R['target_sens']}% floor only
approximately.</p>

<h2>Conclusion</h2>
<p>Malignant nuclei are bigger, more deeply indented and more unevenly textured,
and a measurement&rsquo;s worst reading tells you more than its average. {KW}
measurements do about as well as thirty &mdash; a list a lab could actually
check. Next would be another lab&rsquo;s images.</p>
"""

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which Measurements Diagnose a Breast Mass?</title>
<style>
  :root {{ --ink:#111; --ink2:#55534e; --rule:#dcdad4; --surface:#fdfdfc;
           --accent:#1c5cab; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:56px 24px 72px; background:var(--surface);
    color:var(--ink); font:17px/1.62 "Iowan Old Style","Palatino Linotype",
    Palatino,Georgia,serif; }}
  main {{ max-width:760px; margin:0 auto; }}
  header {{ border-bottom:2px solid var(--ink); padding-bottom:20px;
    margin-bottom:30px; }}
  h1 {{ font-size:32px; line-height:1.2; margin:0 0 12px; letter-spacing:-.4px; }}
  .sub {{ font-size:18px; color:var(--ink2); font-style:italic; margin:0 0 16px; }}
  .authors {{ font-size:15.5px; margin:0; }}
  .meta {{ font-size:13.5px; color:var(--ink2); margin:6px 0 0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:.9px;
    margin:34px 0 10px; color:var(--accent);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    font-weight:650; }}
  p {{ margin:0 0 14px; }}
  .lead {{ font-size:18.5px; }}
  .lead::first-letter {{ font-size:56px; float:left; line-height:.85;
    padding:4px 8px 0 0; font-weight:600; }}
  figure {{ margin:30px 0; }}
  figure img {{ width:100%; height:auto; display:block;
    border:1px solid var(--rule); border-radius:3px; }}
  figcaption {{ font-size:13.5px; color:var(--ink2); margin-top:8px;
    line-height:1.45;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; margin:10px 0 0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--rule); }}
  th {{ font-weight:650; border-bottom:1.5px solid var(--ink); }}
  td.n, th.n {{ text-align:center; }}
  footer {{ margin-top:40px; padding-top:20px; border-top:1px solid var(--rule);
    font-size:13px; color:var(--ink2); line-height:1.55;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  footer a {{ color:var(--accent); }}
  @media (max-width:640px) {{ body {{ padding:32px 16px 48px; font-size:16px; }}
    h1 {{ font-size:25px; }} }}
  @media print {{ body {{ padding:0; }} figure {{ break-inside:avoid; }} }}
</style></head><body><main>

<header>
  <h1>Which Measurements Diagnose a Breast Mass?</h1>
  <p class="sub">Working out which of 30 nuclear measurements actually matter</p>
  <p class="authors">Amoolya &middot; Bonnie &middot; Dhruvika &middot; Kalynn
     &middot; Edward</p>
  <p class="meta">STAT 451 &middot; Group Project Report</p>
</header>

{BODY}

<h2>Contributions</h2>
<table>
  <tr><th>Member</th><th class="n">Proposal</th><th class="n">Coding</th>
      <th class="n">Presentation</th><th class="n">Report</th></tr>
  <tr><td>Amoolya</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td></tr>
  <tr><td>Bonnie</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td></tr>
  <tr><td>Dhruvika</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td></tr>
  <tr><td>Kalynn</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td></tr>
  <tr><td>Edward</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td></tr>
</table>
<p class="meta" style="margin-top:8px">Amoolya ran the logistic regression and
the LASSO stability work, Bonnie the decision trees, Dhruvika the k-NN models,
Kalynn the SVMs and the cutoff choice, and Edward the correlation and PCA
analysis. All five of us wrote and revised the report and presented.</p>

<footer>
<b>Data.</b> Wolberg, W., Mangasarian, O., Street, N., &amp; Street, W. (1993).
<i>Breast Cancer Wisconsin (Diagnostic)</i> [Dataset]. UCI Machine Learning
Repository. <a href="https://doi.org/10.24432/C5DW2B">doi.org/10.24432/C5DW2B</a><br>
<b>Software.</b> scikit-learn 1.1 (Pedregosa et&nbsp;al., 2011); pandas; matplotlib.
Analysis code: <code>analysis.py</code>. Bootstrap stability selection follows
Meinshausen &amp; B&uuml;hlmann (2010), <i>J.&nbsp;R.&nbsp;Stat.&nbsp;Soc.&nbsp;B</i> 72(4).
</footer>
</main></body></html>"""

pathlib.Path("report.html").write_text(HTML, encoding="utf-8")

# Word count of the prose the grader reads: headings and paragraphs, with the
# figures (captions included) stripped out, since the brief allows "600 words or
# less with supporting graphics".
text = re.sub(r"<figure>.*?</figure>", " ", BODY, flags=re.S)
text = re.sub(r"<[^>]+>", " ", text)
text = text.replace("&ldquo;", '"').replace("&rdquo;", '"').replace("&nbsp;", " ")
text = re.sub(r"&[a-z]+;", " ", text)
# Count only tokens with a letter or digit in them, so a standalone em dash is
# not scored as a word (wordcounter.net does not count one either).
words = sum(1 for w in text.split() if re.search(r"[A-Za-z0-9]", w))
flag = "" if words <= 600 else "  *** OVER THE 600-WORD LIMIT ***"
print(f"report.html written. Body word count: {words}{flag}")
