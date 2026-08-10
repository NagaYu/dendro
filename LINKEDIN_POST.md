# LinkedIn post

Primary version below. A Japanese cut, a short version for X / Bluesky, two likely comment
replies, and posting notes follow.

---

## Main post (English, ~340 words)

AI-text detectors read prose. Prose is the part an adversary controls.

Paraphrase a human's writing and a perplexity detector doesn't just fail — in my benchmark it
goes **below chance** (AUC 0.14), systematically calling rewritten human writing synthetic.
Attach a forged `<meta name="date">` and it notices nothing at all, because the prose is
unchanged.

So I built **Dendro** to ask a different question: *who, other than the author, saw this — and
when?*

**Code, benchmark, tests:** https://github.com/NagaYu/dendro
**Dataset:** https://huggingface.co/datasets/NagaYu/dendro-lowbackground

The name is from dendrochronology. You don't date a beam by inspecting the grain for signs of
modernity — you match its ring pattern against an independent chronology. Dendro matches a
document's fingerprint against archives whose timestamps are already witnessed: the Internet
Archive, Common Crawl, arXiv, Crossref, public posting archives.

**Measured on 5,034 real arXiv records** against a perplexity detector (A) and a learned
classifier (B):

• Six generator generations, each **measurably** closer to human text: (A) decays 0.89 → 0.62.
Dendro is flat at 1.00 — it never reads the prose, so there is no channel through which the
generator *could* matter
• On a generator family nobody trained on, (A) inverts to **0.23**. Confidently wrong
• Paraphrase attack strong enough to destroy 83% of exact word 5-grams: (A) 0.14, Dendro 1.00
— the rewrite aligns to its 2019 ancestor and inherits that date
• Backdated metadata: **100% of forgeries caught, 0% false accusations.** (A) and (B) catch 0%,
structurally — they read prose, and the prose is innocent

**And the column that matters most:** on recent human documents that no archive happens to
hold, Dendro drops to **0.500 — exactly chance — and abstains on 100% of them.** The learned
classifier scores 0.99 there. For unarchived recent text, (B) is simply the better tool.

I put that column in the README table, not a footnote. A provenance tool that publishes only
its flattering numbers is the failure mode I was trying to avoid.

Dendro is **not** an AI-writing detector and must not be used as one. It reports archival
evidence of prior existence. A document with no record is *unknown*, never *generated* — most
text that has ever existed was never archived by anyone.

96 tests, fully offline. Apache-2.0.

#DataProvenance #MachineLearning #DataQuality #OpenSource #HuggingFace #AIDetection

---

## The "what didn't work" follow-up post (post 2-3 days later)

Five things that were broken in Dendro before it worked, because deleting them just means the
next person rediscovers them.

**1. Retrieval was blind to the exact attack it existed to survive.**
The LSH index banded only exact word shingles — the channel paraphrase destroys. So a rewrite
whose word-containment had fallen to 0.18 produced no band collision and its true source was
never even *considered*. Classification logic was perfect. Ancestor recall was 25%. Every
component looked correct in isolation; retrieval has to survive whatever the attack is, or the
rest of the pipeline never runs.

**2. My paraphrase "attack" wasn't an attack.**
Synonym substitution only. At maximum strength it still left 54% word-channel containment —
robustness measured against it would have been an artefact of a weak adversary. Real rewriting
*restructures*: insert one token in four and essentially every 5-gram is gone. Rebuilt, it
drops containment to 0.17.

**3. The calibrator was accusing real people.**
Fitted on a split containing pre-2021 human and synthetic text but no *recent* human text, the
isotonic map extrapolated genuine 2025 abstracts straight into the accusatory region:
P(human) ≈ 0.03. Calibration data has to cover the deployment distribution. That one was the
whole ethical claim of the project, quietly false.

**4. A perfect score that measured nothing.**
The learned baseline hit AUC 1.0 on *every* generator generation — not by recognising
generated text, but by recognising the 41-document vocabulary my generator had been trained
on. Scaled the generator's training corpus to 900 documents and the axis became real.

**5. A headline number that was a corpus artefact.**
Dendro scored ~1.0 separating recent human text from synthetic — because every human document
in that split was an arXiv paper with a registration record and every synthetic one had none
*by construction*. A one-line `has_witness` check scores identically. I added a control
condition that strips the witnesses; Dendro falls to chance and abstains, which is the honest
answer.

Also: an ECE of 0.000 is not a triumph. On a perfectly separable task an isotonic map collapses
to a step function. It reflects the task, not the method.

https://github.com/NagaYu/dendro

---

## 日本語版（~400字）

AI生成テキスト検出器は「文章」を読みます。しかし文章は、攻撃者が自由に書き換えられる部分です。

人間が書いた文章をLLMで言い換えると、パープレキシティ系検出器は失敗するだけでなく、**AUC 0.14
——偶然以下**まで落ちます。つまり人間の文章を体系的に「合成」と誤判定する。逆に、偽の古い日付を
メタデータに埋め込んでも何も気づきません。文章自体は変わっていないからです。

そこで **Dendro** は別の問いを立てます。「著者以外の誰が、いつこの内容を見たか?」

年輪年代学が由来です。木材の年代は木目を見て判定するのではなく、既知の年代系列と照合して決めます。
Dendroも同様に、Internet Archive・Common Crawl・arXiv 等、**独立した運営主体**のアーカイブ記録と
文書指紋を照合します。

実データ(arXiv 5,034件)での測定:

• 生成モデルの世代が上がるほど (A) は 0.89 → 0.62 に劣化。Dendro は 1.00 で水平——本文を一切
読まないため、生成モデルが影響する経路が存在しない
• 未知の生成モデル系列では (A) は **0.23** に反転(自信を持って誤答)
• 言い換え攻撃(単語5-gramの83%を破壊)でも Dendro は 1.00。2019年の祖先文書に整列し、その
日付を継承する
• バックデート偽装:**検出率100%、誤検出0%**。(A)(B) は構造上0%

**そして最も重要な数字**:アーカイブに記録のない最近の人間の文章では、Dendro は **0.500——
完全に偶然——となり、100%棄権します**。学習型検出器はそこで0.99。未アーカイブの最近の文章に
関しては (B) の方が優れたツールです。

この列を脚注ではなくREADMEの表に入れました。都合の良い数字だけを公開する来歴ツールこそ、
避けたかった失敗そのものだからです。

Dendro は AI検出器ではなく、そう使ってはいけません。報告するのは「事前存在の証跡」だけです。
記録がない文書は「不明」であって「生成」ではありません。

https://github.com/NagaYu/dendro

---

## Short version (X / Bluesky)

AI-text detectors read prose. Prose is what an adversary controls.

Paraphrase human writing → perplexity detector goes *below chance* (0.14), calling humans
synthetic. Forge a date → it notices nothing.

Dendro asks who else saw the document, and when. Flat 1.00 across six generator generations,
100% of backdate forgeries caught.

And 0.500 — chance — where no archive holds a record. It abstains instead of guessing.

https://github.com/NagaYu/dendro

---

## Comment reply: "isn't AUC 1.00 just overfitting / too good to be true?"

Fair, and the honest answer is that 1.00 is measuring something narrower than it looks.

In the benchmark, pre-2021 documents carry a real arXiv registration record and generated
documents carry none. Dendro separates those perfectly — but so would a one-line `has_witness`
check. The 1.00 is a property of "does an independent archive hold this content", which is
exactly what Dendro claims to measure and *not* a claim about detecting synthesis.

That's why the table also reports the control: same recent human documents, witnesses stripped.
Dendro → 0.500 and abstains on 100%. Learned classifier → 0.990. Where there is no evidence,
an evidence-based method has nothing, and the learned detector is the better tool.

The interesting claim isn't the 1.00. It's that the 1.00 doesn't move when you change the
generator, paraphrase the text, or forge the metadata — because none of those touch the input.

---

## Comment reply: "why not just trust the git commit date / the metadata?"

Because both are author-controlled.

`GIT_AUTHOR_DATE="2019-03-01" git commit` — two environment variables and a file "existed" in
2019. Dendro models a git date at forgeability 0.5 and an Internet Archive capture at 1e-3, and
that gap is the whole design. The Wayback Machine will let you create a capture *now*; it will
not let you create one dated 2019.

The useful asymmetry is that forgery leaves contradictions. A commit claiming 2019 inside a
repository GitHub says was created in 2025 is a contradiction the forger can't fix, because the
repo creation timestamp lives on GitHub's servers and not in the object graph.

And the restraint matters as much: `log LR = −k·log(1−c)` goes to exactly *zero* as archive
coverage `c` goes to zero. No coverage measurement, no accusation. A page nobody ever crawled
is never suspected merely for being obscure.

---

## Notes for posting

- **No Space link yet.** Hugging Face now requires PRO to host Gradio Spaces on free CPU. The
  app is built and one command from publishing (`python -m scripts.publish_space --push`). If
  you subscribe, add this line under the Code link and it becomes the strongest hook:
  `**Try it:** https://huggingface.co/spaces/NagaYu/dendro`
- **Image:** use `figures/fig1_headline.png`. The flat teal line against the collapsing red one
  reads at thumbnail size, and the right-hand panel pre-empts "how do you know the generations
  got better?" — that axis is measured, not asserted.
  Alternative: `figures/fig2_robustness.png`, whose left panel crossing *below* the chance line
  is the single most arresting result in the project.
- LinkedIn strips most formatting. The bullets use `•` and the arrows `→` so they survive.
- The "what didn't work" post is the one that will travel furthest with engineers. Give the
  main post 2-3 days first so it doesn't split attention.
- If someone pushes back that this only works for archived content — agree immediately. That's
  the binding constraint, it's in the README's limitations section, and archived text
  over-represents the English institutional indexed web. It's a fairness problem, not a footnote.
