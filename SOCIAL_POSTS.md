# Social posts — LinkedIn and X

**Paste-ready.** LinkedIn renders no markdown — asterisks show up literally — so the two main
versions below are plain text, using only `•`, `→` and `—`, which survive. Copy from the first
line to the hashtags and paste directly.

A short social cut, two likely comment replies, and posting notes follow.

---

## Main post — English (paste-ready, ~330 words)

AI-text detectors read prose. Prose is the part an adversary controls.

Paraphrase someone's writing and a perplexity detector doesn't merely fail. In my benchmark it
goes below chance — AUC 0.14 — systematically calling rewritten human writing synthetic. Forge
the date in a meta tag and it notices nothing at all, because the prose is innocent.

So I built Dendro to ask a different question: who, other than the author, saw this — and when?

Code, benchmark, tests → https://github.com/NagaYu/dendro
Dataset → https://huggingface.co/datasets/NagaYu/dendro-lowbackground

The name is from dendrochronology. You don't date a beam by inspecting the grain for signs of
modernity. You match its ring pattern against an independent chronology built from wood whose
age is already known. Dendro matches a document's fingerprint against archives whose timestamps
are already witnessed — the Internet Archive, Common Crawl, arXiv, Crossref, public posting
archives — and treats operators, not records, as the unit of independence. Twenty captures from
one archive are one archive.

Measured on 5,034 real arXiv records, against a perplexity detector (A) and a learned
classifier (B):

• Six generator generations, each measurably closer to human text. (A) decays 0.89 → 0.62.
Dendro stays flat at 1.00 — it never reads the prose, so there is no channel through which the
generator could matter.

• On a generator family nobody trained on, (A) inverts to 0.23. Not uninformative. Confidently
wrong.

• A paraphrase attack strong enough to destroy 83% of exact word 5-grams: (A) 0.14, Dendro
1.00. The rewrite aligns to its 2019 ancestor and inherits that date.

• Backdated metadata: 100% of forgeries caught, 0% false accusations. (A) and (B) catch 0% —
structurally, because they read prose and the prose is unchanged.

And the number that matters most. On recent human documents that no archive happens to hold,
Dendro drops to 0.500 — exactly chance — and abstains on 100% of them. The learned classifier
scores 0.99 there. For unarchived recent text, (B) is simply the better tool.

That column is in the README table, not a footnote. A provenance tool that publishes only its
flattering numbers is the failure mode I was trying to avoid.

Dendro is not an AI-writing detector and must not be used as one. It reports archival evidence
of prior existence. A document with no record is unknown, never generated — most text that has
ever existed was never archived by anyone.

96 tests, fully offline, byte-identical across runs. Apache-2.0.

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

## Main post — 日本語（paste-ready）

AI生成テキスト検出器は「文章」を読みます。しかし文章は、攻撃者が自由に書き換えられる部分です。

人間が書いた文章をLLMで言い換えると、パープレキシティ系の検出器は単に失敗するのではなく、
AUC 0.14——偶然を下回ります。つまり人間の文章を、体系的に「合成」と誤判定する。逆に、メタタグに
偽の古い日付を埋め込んでも何も気づきません。文章自体は変わっていないからです。

そこで Dendro は別の問いを立てました。著者以外の誰が、いつこの内容を見たのか?

コード・ベンチマーク・テスト → https://github.com/NagaYu/dendro
データセット → https://huggingface.co/datasets/NagaYu/dendro-lowbackground

名前は年輪年代学に由来します。木材の年代は、木目に「新しさの兆候」を探して判定するのではなく、
年代が既知の木材から作られた独立した年輪系列と照合して決めます。Dendro も同じように、文書の指紋を
Internet Archive・Common Crawl・arXiv・Crossref 等、既に時刻が witness されたアーカイブと照合
します。そして独立性の単位を「レコード数」ではなく「運営主体」に置きます。同一アーカイブの20件の
キャプチャは、1つのアーカイブでしかありません。

実データ(arXiv 5,034件)での測定。比較対象はパープレキシティ系検出器 (A) と学習型分類器 (B):

• 生成モデルの世代を6段階、各世代が「人間の文章への距離」で実測して近づく設計。(A) は
0.89 → 0.62 に劣化。Dendro は 1.00 で水平——本文を一切読まないため、生成モデルが影響しうる
経路が存在しない。

• 学習時に存在しなかった生成モデル系列では、(A) は 0.23 に反転。情報がないのではなく、
自信を持って誤答する。

• 単語5-gramの83%を破壊する言い換え攻撃:(A) 0.14、Dendro 1.00。書き換え後も2019年の祖先文書に
整列し、その日付を継承する。

• バックデート偽装:検出率100%、誤検出0%。(A)(B) は構造上0%——本文を読む手法にとって、
本文は無改変だから。

そして最も重要な数字。アーカイブに記録が存在しない最近の人間の文章では、Dendro は 0.500——
完全に偶然——まで落ち、100%棄権します。学習型分類器はそこで 0.99。未アーカイブの最近の文章に
関しては、(B) の方が単純に優れたツールです。

この列は脚注ではなく README の表に入れました。都合の良い数字だけを公開する来歴ツールこそ、
避けたかった失敗そのものだからです。

Dendro は AI検出器ではありませんし、そう使ってはいけません。報告するのは「事前存在の証跡」
だけです。記録のない文書は「不明」であって「生成」ではありません——これまで存在した文章の
大半は、誰にもアーカイブされていないのですから。

テスト96件、完全オフライン、再実行でバイト単位一致。Apache-2.0。

#データ来歴 #機械学習 #データ品質 #OpenSource #HuggingFace #AI検出

---

## X / Bluesky

Every post below is verified against X's *weighted* character count — CJK counts double
(so Japanese effectively gets 140), and URLs always count as 23 regardless of length. Counts
in brackets. Paste each block as one post.

### Single standalone post — English [272/280]

AI-text detectors read prose. Prose is what an adversary controls.

Paraphrase human writing and a perplexity detector doesn't just fail — it goes below chance
(AUC 0.14), calling humans synthetic.

Dendro asks who else saw the document, and when.

https://github.com/NagaYu/dendro

---

### Thread — English (7 posts)

**1/7 [268]**

AI-text detectors read prose.

Prose is the part an adversary controls.

Paraphrase someone's writing and a perplexity detector doesn't merely fail — in my benchmark it
goes below chance, AUC 0.14, systematically calling human writing synthetic.

So I built Dendro. 🧵

**2/7 [239]**

It asks a different question: who, other than the author, saw this — and when?

The name is from dendrochronology. You don't date a beam by inspecting the grain for signs of
modernity. You match its rings against an independent chronology.

**3/7 [243]**

Dendro matches a document's fingerprint against archives that already witnessed it: Internet
Archive, Common Crawl, arXiv, Crossref.

The unit of independence is the *operator*, not the record. Twenty captures from one archive
are one archive.

**4/7 [278]**

Six generator generations, each measurably closer to human text.

Perplexity detector: 0.89 → 0.62
Dendro: flat at 1.00

It never reads the prose, so the generator has no channel to matter through. On an unseen
generator family the detector inverts to 0.23 — confidently wrong.

**5/7 [222]**

Backdated metadata is the one that surprised me.

Forge a date in a meta tag: 100% of forgeries caught, 0% false accusations.

Both text-based baselines catch 0% — structurally. They read prose, and the prose is unchanged.

**6/7 [278]**

And the number that matters most.

On recent human documents no archive holds, Dendro drops to 0.500 — exactly chance — and
abstains on 100% of them. The learned classifier scores 0.99 there.

For unarchived text it's the better tool. That's in the README table, not a footnote.

**7/7 [267]**

Dendro is NOT an AI-writing detector and must not be used as one. It reports archival evidence
of prior existence. No record = unknown, never "generated".

5,034 real arXiv records. 96 tests, fully offline. Apache-2.0.

https://github.com/NagaYu/dendro
https://huggingface.co/datasets/NagaYu/dendro-lowbackground

---

### スレッド — 日本語（7投稿）

**1/7 [237]**

AI生成テキスト検出器は「文章」を読む。しかし文章は、攻撃者が自由に書き換えられる部分だ。

人間の文章をLLMで言い換えると、パープレキシティ系検出器はAUC 0.14——偶然を下回る。人間の文章を体系的に「合成」と誤判定する。

そこでDendroを作った🧵

**2/7 [171]**

問いを変える。「著者以外の誰が、いつこの内容を見たか?」

名前は年輪年代学から。木材の年代は木目に新しさの兆候を探すのではなく、年代既知の独立した年輪系列と照合して決める。

**3/7 [229]**

Dendroは文書の指紋を、既に時刻がwitnessされたアーカイブと照合する。Internet Archive、Common Crawl、arXiv、Crossref。

独立性の単位は「レコード数」ではなく「運営主体」。同一アーカイブの20件のキャプチャは1つのアーカイブでしかない。

**4/7 [237]**

生成モデル6世代、各世代が「人間の文章への距離」で実測して近づく設計。

パープレキシティ系: 0.89 → 0.62
Dendro: 1.00 で水平

本文を一切読まないため、生成モデルが影響しうる経路が存在しない。未知の生成モデル系列では検出器は0.23に反転する。

**5/7 [188]**

バックデート偽装が一番意外だった。

メタタグに偽の古い日付を埋め込む → 検出率100%、誤検出0%。

本文ベースの手法は両方とも0%。構造上の話だ。本文を読む手法にとって、本文は無改変なのだから。

**6/7 [239]**

そして最も重要な数字。

アーカイブに記録のない最近の人間の文章では、Dendroは0.500——完全に偶然——まで落ち、100%棄権する。学習型分類器はそこで0.99。

未アーカイブの文章に関しては学習型の方が優れている。この列は脚注ではなくREADMEの表に入れた。

**7/7 [218]**

DendroはAI検出器ではないし、そう使ってはいけない。報告するのは「事前存在の証跡」だけ。記録がない=「不明」であって「生成」ではない。

arXiv実データ5,034件。テスト96件、完全オフライン。Apache-2.0

https://github.com/NagaYu/dendro

---

### X posting notes

- **Attach the image to post 1**, not the last one. `figures/fig2_robustness.png` for the
  thread — its left panel crossing *below* the chance line is exactly what post 1 claims, and
  a reader who only sees the first post still gets the finding. Use
  `figures/fig1_headline.png` for the single standalone post instead, since that one leads on
  the flat line rather than the paraphrase result.
- **Links in the last post only.** X suppresses reach on posts containing links; keeping 1/7
  link-free gives the thread a better chance of being seen at all.
- Post 6 is the one that earns credibility with researchers. Don't cut the thread at 5 to save
  effort — ending on "100% caught" without the abstention result reads as marketing.
- If you get HF PRO and I push the Space, add `Try it → https://huggingface.co/spaces/NagaYu/dendro`
  to post 7. It fits: 7/7 is at 267 with two URLs, and a third would push it over, so drop the
  dataset link and keep the Space + repo.
- Verify any edit with `python3 scratchpad/xcount.py`-style weighting — CJK doubles, so the
  Japanese posts have far less headroom than the character count in an editor suggests.

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
